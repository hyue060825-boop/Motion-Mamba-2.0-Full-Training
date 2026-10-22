"""Full-clip controller development on three metadata-selected BONES dev pairs.

Official G1 references only. No HML/HiPHI retarget, held-out access, policy
tuning on acceptance, or fusion release is performed by this diagnostic.
"""
import io
import json
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT,BONES,read_jsonl,write_json,sha,digest,atomic_bytes
from g1_pipeline.gates import require_pass
from g1_pipeline.scene_policy import classify
from g1_pipeline.sonic_scene import prepare_scene,torque_control
from g1_pipeline.sonic_observations import SonicObservations
from g1_pipeline.robot import read_csv,FORMAT
from g1_pipeline.resample import qpos as resample


def run():
    require_pass('00_preflight')
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';rows=read_jsonl(manifest);selected=[]
    for category in ('walking','turning','crouching'):
        candidates=[r for r in rows if classify(r)['core_eligible'] and category in str(r.get('category')) and r['sample_id'] not in [x['sample_id'] for x in selected]]
        if not candidates:raise ValueError('MISSING_CONTROLLER_DEVELOPMENT_CATEGORY:'+category)
        row=min(candidates,key=lambda r:digest({'seed':20260915,'sample_id':r['sample_id']}));selected.append({**row,'tracking_category':category})
    out=ROOT/'work/sonic_bones_dev3_v1'
    if out.exists():raise ValueError('CONTROLLER_DIAGNOSTIC_NAMESPACE_EXISTS')
    out.mkdir()
    design={'status':'CONTROLLER_DEVELOPMENT_ONLY','input_manifest_sha256':sha(manifest),'selected':[{'sample_id':r['sample_id'],'category':r['tracking_category']} for r in selected],
            'selection':'metadata-only flat core; walking/turning/crouching; minimum fixed seeded ID hash; no motion-quality selection',
            'seeds':[0,1,2],'control_dt_s':.02,'initialization':'reference first qpos and differentiable qvel; joint noise std 0.002 rad; no root height adjustment',
            'termination':'nonfinite or root height loss >0.35m from contemporaneous reference; all other errors measured without pass threshold',
            'controller_gate':'NOT_FROZEN','code_sha256':sha(__file__),'resources':{'workers':1,'cpu_threads':1,'gpu':False}}
    write_json(out/'design.json',design)
    sys.path.insert(0,str(ROOT/'deps/onnxruntime_1_22_1'))
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    interface=SonicObservations();sessions=[]
    for item in interface.audit['models']:
        if sha(item['path'])!=item['sha256']:raise ValueError('SONIC_MODEL_HASH_CHANGED')
        sessions.append(ort.InferenceSession(item['path'],sess_options=options,providers=['CPUExecutionProvider']))
    model,scene=prepare_scene();dt=.02;substeps=round(dt/model.opt.timestep);results=[]
    if abs(substeps*model.opt.timestep-dt)>1e-10:raise ValueError('INCOMPATIBLE_SIM_DT')
    feet=[model.body(n).id for n in ['left_ankle_roll_link','right_ankle_roll_link']]
    for row in selected:
        path=BONES/row['source_paths']['g1'];native,nt,_=read_csv(path,format_id=FORMAT,fps=row['original_fps'])
        q25,t25,_=resample(native,nt,model.jnt_range[1:])
        ref,times,time_map=resample(q25,t25,model.jnt_range[1:],fps=50.)
        rv=np.empty((len(ref),35))
        for i in range(len(ref)-1):mujoco.mj_differentiatePos(model,rv[i],dt,ref[i],ref[i+1])
        rv[-1]=rv[-2]
        rd=mujoco.MjData(model);ref_feet=[]
        for q in ref:rd.qpos[:]=q;mujoco.mj_kinematics(model,rd);ref_feet.append(rd.xpos[feet].copy())
        ref_feet=np.array(ref_feet)
        for seed in (0,1,2):
            obs=SonicObservations();data=mujoco.MjData(model);data.qpos[:]=ref[0];data.qvel[:]=rv[0]
            data.qpos[7:]+=np.random.default_rng(seed).normal(0,.002,29);mujoco.mj_forward(model,data)
            poses=[data.qpos.copy()];actual_feet=[data.xpos[feet].copy()];last=np.zeros(29);saturation=[];failure=None;peak_speed=0.
            for f in range(len(ref)-1):
                spatial=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,model.body('pelvis').id,spatial,1)
                obs.push_state(data.qpos[7:],data.qvel[6:],data.qpos[3:7],spatial[:3],last,time_s=float(times[f]))
                token=sessions[0].run(None,{'obs_dict':obs.encoder(ref[:,7:],rv[:,6:],ref[:,3:7],f,data.qpos[3:7],timestamps=times)})[0]
                last=sessions[1].run(None,{'obs_dict':obs.decoder(token)})[0].reshape(29);target=obs.joint_target(last)
                for _ in range(substeps):
                    torque,info=torque_control(model,data.qpos,data.qvel,target,obs.constants);data.ctrl[:]=torque;saturation.append(info['saturated']);mujoco.mj_step(model,data)
                    peak_speed=max(peak_speed,float(np.max(abs(data.qvel[6:]))))
                mujoco.mj_kinematics(model,data);poses.append(data.qpos.copy());actual_feet.append(data.xpos[feet].copy())
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():failure='NONFINITE_SIMULATION';break
                if data.qpos[2]<ref[f+1,2]-.35:failure='ROOT_HEIGHT_LOSS_OVER_0_35M';break
            q=np.array(poses);r=ref[:len(q)];rot=np.rad2deg((R.from_quat(r[:,3:7],scalar_first=True).inv()*R.from_quat(q[:,3:7],scalar_first=True)).magnitude())
            p=out/(row['source_motion_id']+f'.seed{seed}.npz');buf=io.BytesIO();np.savez(buf,qpos=q,reference=r,timestamps=times[:len(q)],saturation=np.array(saturation));atomic_bytes(p,buf.getvalue())
            result={'sample_id':row['sample_id'],'category':row['tracking_category'],'seed':seed,'full_duration_completed':failure is None,
                    'failure':failure,'reference_control_steps':len(ref)-1,'completed_control_steps':len(q)-1,'reference_duration_s':float(times[-1]-times[0]),
                    'joint_rmse_rad':float(np.sqrt(np.mean((q[:,7:]-r[:,7:])**2))),
                    'root_translation_rmse_m':float(np.sqrt(np.mean(np.sum((q[:,:3]-r[:,:3])**2,axis=1)))),
                    'root_orientation_mean_deg':float(rot.mean()),'root_orientation_max_deg':float(rot.max()),
                    'feet_position_mean_m':np.linalg.norm(np.array(actual_feet)-ref_feet[:len(q)],axis=-1).mean(axis=0).tolist(),
                    'torque_saturation_fraction':float(np.mean(saturation)),'joint_speed_max_rad_s':peak_speed,
                    'trajectory_sha256':sha(p),'source_sha256':sha(path),'release_status':'NOT_EVALUATED'}
            results.append(result);print(result,flush=True)
            write_json(ROOT/'audit/sonic_bones_dev3_v1.json',{'design':design,'scene':scene,'results':results,
                       'three_source_dynamic_gate':False,'source':'BONES official development references only','status':'DEVELOPMENT_MEASUREMENTS'})
    return results


if __name__=='__main__':run()
