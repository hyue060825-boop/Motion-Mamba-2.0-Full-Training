"""Eight largest development rate violations, full clips and full paired QA."""
from motion_mamba2.paths import artifact_path
import ast
import io
import json
import re
from pathlib import Path
import numpy as np
from g1_pipeline.common import ROOT,BONES,read_jsonl,write_json,sha,atomic_bytes
from g1_pipeline.gates import require_pass
from g1_pipeline.robot import compile_target,read_csv,FORMAT
from g1_pipeline.resample import qpos as resample
from g1_pipeline.bounded_hinge_trajectory import project,sonic_training_reference_limits
from g1_pipeline.bones_continuity_qa import metrics as continuity
from g1_pipeline.bones_pair_qa import run as pair_qa
from g1_pipeline.summarize_bones_qa import summarize


def run():
    require_pass('00_preflight')
    namespace='bones_rate_projection_v20';out=ROOT/'work'/namespace
    if out.exists():raise ValueError('DIAGNOSTIC_NAMESPACE_EXISTS')
    out.mkdir()
    model,asset=compile_target();limits=sonic_training_reference_limits(model)
    controller_source=Path(artifact_path('g1_py_9291fdfac6'))
    tree=ast.parse(controller_source.read_text());expressions=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ImplicitActuatorCfg':
            kw={k.arg:k.value for k in node.keywords}
            names=ast.literal_eval(kw['joint_names_expr']);rates=ast.literal_eval(kw['velocity_limit_sim'])
            expressions.extend((name,rates[name] if isinstance(rates,dict) else rates) for name in names)
    for j,value in enumerate(limits,1):
        matches=[v for pattern,v in expressions if re.fullmatch(pattern,model.joint(j).name)]
        if matches!=[value]:raise ValueError('SONIC_TRAINING_CONFIG_RATE_MISMATCH')
    base='bones_temporal_arms_dev_v15';rp=ROOT/'qa/aggregate'/f'{base}.json'
    base_rows=json.loads(rp.read_text())['results']
    selected=sorted(base_rows,key=lambda r:(-float(np.max(np.array(r['candidate_continuity']['hinge_speed_max_rad_s'])/limits)),r['sample_id']))[:8]
    meta={r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')}
    design={'panel':'dev','status':'DEVELOPMENT_RATE_TRANSFORM_ONLY','expected_panel':200,'selected_clips':8,
            'selection':'eight largest per-joint speed / SONIC training config limit ratios from v15 dev200',
            'sample_ids':[r['sample_id'] for r in selected],'limits_rad_s':limits.tolist(),
            'correction_curvature':1.,'dt_s':.04,'reference_input_to_optimizer':False,
            'hardware_claim':False,'limitation':'SONIC training model differs from locked target in some motor effort limits; these are experimental reference rate bounds only',
            'hashes':{str(p):sha(p) for p in [rp,controller_source,ROOT/'g1_pipeline/bounded_hinge_trajectory.py',ROOT/'scripts/diagnose_bones_rate_projection.py']},
            'asset_hash':asset['asset_hash'],'resources':{'workers':1,'gpu':False}}
    write_json(out/'run.json',design);results=[]
    for bound in selected:
        row=meta[bound['sample_id']];mid=row['source_motion_id']
        qp=ROOT/'work'/base/(mid+'.npy');tp=qp.with_suffix('.time.json')
        if sha(qp)!=bound['output_sha256'] or sha(tp)!=bound['time_sha256']:raise ValueError('BASE_HASH_CHANGED')
        q=np.load(qp);t=np.asarray(json.loads(tp.read_text())['source']['target_timestamps'])
        corrected=q.copy();corrected[:,7:],info=project(q[:,7:],model.jnt_range[1:,0],model.jnt_range[1:,1],limits)
        op=out/(mid+'.npy');buf=io.BytesIO();np.save(buf,corrected);atomic_bytes(op,buf.getvalue())
        atomic_bytes(out/(mid+'.time.json'),tp.read_bytes());write_json(out/(mid+'.solver.json'),info)
        reference,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
        reference,_,_=resample(reference,rt,model.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
        error=np.rad2deg(abs(corrected[:,7:]-reference[:,7:]))
        result={**bound,'hinge_mae_deg':float(error.mean()),'per_joint_mae_deg':error.mean(axis=0).tolist(),
                'candidate_continuity':continuity(model,corrected,t),'output_sha256':sha(op),
                'solver_sha256':sha(out/(mid+'.solver.json')),'rate_projection':info}
        results.append(result);print(bound['sample_id'],max(bound['candidate_continuity']['hinge_speed_max_rad_s']),
                                   max(result['candidate_continuity']['hinge_speed_max_rad_s']),result['hinge_mae_deg'],flush=True)
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',{'status':'DEVELOPMENT_SUBSET','fixed_panel_size':200,
               'actual_attempted':len(results),'run_config':design,'results':results})
    pair_qa(namespace,'dev',sample_ids=design['sample_ids']);summarize(namespace)


if __name__=='__main__':run()
