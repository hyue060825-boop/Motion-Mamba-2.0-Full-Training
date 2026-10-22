"""Read-only SONIC observation/model/motor audit; no inference or robot I/O."""
from motion_mamba2.paths import artifact_path
import ast
import json
import re
import subprocess
from pathlib import Path
import onnx
import yaml
from g1_pipeline.common import ROOT,write_json,sha,atomic_bytes
from g1_pipeline.robot import compile_target


def run():
    repo=Path(artifact_path('gr00t-wholebodycontrol_cb55417b96'))
    base=repo/'gear_sonic_deploy'
    source=base/'src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp'
    header=base/'src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp'
    config=base/'policy/release/observation_config.yaml'
    native=repo/'gear_sonic/envs/manager_env/robots/g1.py'
    text=source.read_text();cfg=yaml.safe_load(config.read_text())
    dimensions={name:int(dim) for name,dim in re.findall(r'\{"([a-z0-9_]+)", (\d+), \[this\]',text)}
    dimensions['token_state']=int(cfg['encoder']['dimension'])
    def layout(items):
        offset=0;rows=[]
        for item in items:
            if not item.get('enabled',True):continue
            name=item['name'];width=dimensions[name]
            rows.append({'name':name,'start':offset,'stop':offset+width,'width':width});offset+=width
        return rows,offset
    policy,pdim=layout(cfg['observations']);encoder,edim=layout(cfg['encoder']['encoder_observations'])
    models=[]
    for name,expected in [('model_encoder.onnx',edim),('model_decoder.onnx',pdim)]:
        path=base/'policy/release'/name;model=onnx.load(path,load_external_data=False)
        def info(value):return {'name':value.name,'dtype':value.type.tensor_type.elem_type,'shape':[d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]}
        inputs=[info(v) for v in model.graph.input];outputs=[info(v) for v in model.graph.output]
        if inputs[0]['shape']!=[1,expected]:raise ValueError('SONIC_MODEL_OBSERVATION_WIDTH_MISMATCH')
        models.append({'path':str(path),'sha256':sha(path),'inputs':inputs,'outputs':outputs})
    work=ROOT/'work/sonic_parameter_probe';work.mkdir(exist_ok=True)
    names=['isaaclab_to_mujoco','mujoco_to_isaaclab','g1_action_scale','kps','kds','default_angles']
    cpp='#include <vector>\n#include <iostream>\n#include <iomanip>\n#include "'+str(header)+'"\n'
    cpp+='int main(){std::cout<<std::setprecision(17)<<"{";\n'
    for i,name in enumerate(names):
        cpp+='std::cout<<"'+(',' if i else '')+'\\"'+name+'\\":["; for(int j=0;j<29;++j){if(j)std::cout<<",";std::cout<<'+name+'[j];} std::cout<<"]";\n'
    cpp+='std::cout<<"}\\n";}\n'
    probe=work/'probe.cpp';atomic_bytes(probe,cpp.encode())
    subprocess.run(['g++','-std=c++17','-O2',str(probe),'-o',str(work/'probe')],check=True)
    constants=json.loads(subprocess.check_output([str(work/'probe')],text=True))
    m,_=compile_target();joint_names=[m.joint(i).name for i in range(1,m.njnt)]
    # Inspect the IsaacLab config without importing IsaacLab or initializing CUDA.
    velocity={}
    for node in ast.walk(ast.parse(native.read_text())):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='ImplicitActuatorCfg':continue
        kwargs={k.arg:k.value for k in node.keywords}
        if 'velocity_limit_sim' not in kwargs:continue
        patterns=ast.literal_eval(kwargs['joint_names_expr']);limits=ast.literal_eval(kwargs['velocity_limit_sim'])
        for name in joint_names:
            if any(re.fullmatch(pattern,name) for pattern in patterns):
                value=next(v for k,v in limits.items() if re.fullmatch(k,name)) if isinstance(limits,dict) else limits
                if name in velocity and velocity[name]!=value:raise ValueError('CONFLICTING_CONTROLLER_SIM_LIMITS:'+name)
                velocity[name]=float(value)
    if set(velocity)!=set(joint_names):raise ValueError('MISSING_JOINT_SIM_VELOCITY_LIMITS')
    # These are controller-training simulator limits, not verified hardware limits.
    report={'status':'STATIC_CONTROLLER_INTERFACE_AUDIT','encoder_layout':encoder,'encoder_width':edim,
            'decoder_layout':policy,'decoder_width':pdim,'encoder_modes':cfg['encoder']['encoder_modes'],
            'models':models,'compiled_deployment_constants':constants,'joint_order_mujoco':joint_names,
            'controller_training_sim_velocity_limits_rad_s':velocity,
            'control_dt_s':.02,'control_dt_evidence':'g1_deploy_onnx_ref.cpp constructor control_dt_(0.02)',
            'anchor_6d':'first two columns of relative rotation, flattened row-major',
            'g1_mode':'absolute reference joint positions/velocities in IsaacLab order; future 10 frames step5; non-required mode inputs zero',
            'policy_joint_positions':'measured minus default_angles, then IsaacLab order; history oldest first',
            'hardware_envelope_verified':False,'inference_run':False,'dynamic_qa_run':False,
            'limitations':['target/controller rest transforms differ and require explicit actuated-scene compatibility',
                           'training simulator speed limits are not proof of deployed speed enforcement',
                           'historical Python simulator effort limits differ from newer C++ policy constants; do not mix unverified configs',
                           'headless observation/inference/actuation equivalence and full tracking protocol remain required'],
            'source_hashes':{str(p):sha(p) for p in [source,header,config,native,probe,Path(__file__)]}}
    write_json(ROOT/'audit/sonic_controller_static_interface.json',report)
    print({'encoder_width':edim,'decoder_width':pdim,'motor_joints':len(velocity),'dynamic_qa_run':False})


if __name__=='__main__':run()
