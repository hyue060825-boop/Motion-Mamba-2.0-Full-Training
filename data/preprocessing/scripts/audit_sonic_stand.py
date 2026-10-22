"""Three-seed synthetic standing check of SONIC on the locked actuated scene.

No dataset motion is used; this is controller integration development, not
the required stratified full-clip three-source dynamic QA.
"""
import io
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT,sha,write_json,atomic_bytes
from g1_pipeline.gates import require_pass
from g1_pipeline.sonic_scene import prepare_scene,torque_control
from g1_pipeline.sonic_observations import SonicObservations


def run():
    require_pass('00_preflight')
    sys.path.insert(0,str(ROOT/'deps/onnxruntime_1_22_1'))
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    interface=SonicObservations();sessions=[]
    for entry in interface.audit['models']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('SONIC_MODEL_HASH_CHANGED')
        sessions.append(ort.InferenceSession(entry['path'],sess_options=opts,providers=['CPUExecutionProvider']))
    model,scene=prepare_scene();dt=.02;steps=250;times=np.arange(steps+1)*dt
    substeps=round(dt/model.opt.timestep)
    if abs(substeps*model.opt.timestep-dt)>1e-10:raise ValueError('NONINTEGER_CONTROLLER_SIM_RATIO')
    neutral=model.qpos0.copy();neutral[7:]=interface.constants['default_angles'];data=mujoco.MjData(model)
    data.qpos[:]=neutral;mujoco.mj_forward(model,data)
    lowest=[]
    for g in range(model.ngeom):
        if '_foot' in (model.geom(g).name or '') and 'collision' in model.geom(g).name:
            if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_CAPSULE:raise ValueError('UNEXPECTED_SYNTHETIC_STAND_FOOT_GEOMETRY')
            radius,half_length=model.geom_size[g,:2];axis_z=data.geom_xmat[g].reshape(3,3)[2,2]
            lowest.append(data.geom_xpos[g,2]-abs(axis_z)*half_length-radius)
    neutral[2]+=.001-min(lowest)
    reference=np.tile(neutral,(len(times),1));velocities=np.zeros((len(times),29));results=[]
    out=ROOT/'work/sonic_synthetic_stand_v1'
    if out.exists():raise ValueError('SYNTHETIC_STAND_NAMESPACE_EXISTS')
    out.mkdir()
    for seed in (0,1,2):
        obs=SonicObservations();data=mujoco.MjData(model);data.qpos[:]=neutral
        data.qpos[7:]+=np.random.default_rng(seed).normal(0,.002,29)
        mujoco.mj_forward(model,data);last_action=np.zeros(29);poses=[data.qpos.copy()];saturation=[];failed=None;max_speed=0.
        for frame in range(steps):
            # MuJoCo object velocity in pelvis local coordinates agrees with
            # the body-local IMU gyro convention used by SONIC.
            spatial=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,model.body('pelvis').id,spatial,1)
            obs.push_state(data.qpos[7:],data.qvel[6:],data.qpos[3:7],spatial[:3],last_action,time_s=times[frame])
            encoder=obs.encoder(reference[:,7:],velocities,reference[:,3:7],frame,data.qpos[3:7],timestamps=times)
            token=sessions[0].run(None,{'obs_dict':encoder})[0]
            action=sessions[1].run(None,{'obs_dict':obs.decoder(token)})[0].reshape(29)
            target=obs.joint_target(action);last_action=action
            for _ in range(substeps):
                torque,info=torque_control(model,data.qpos,data.qvel,target,obs.constants)
                data.ctrl[:]=torque;saturation.append(info['saturated']);mujoco.mj_step(model,data)
                max_speed=max(max_speed,float(np.max(abs(data.qvel[6:]))))
            poses.append(data.qpos.copy())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():failed='NONFINITE_SIMULATION';break
            tilt=np.arccos(np.clip(R.from_quat(data.qpos[3:7],scalar_first=True).as_matrix()[2,2],-1,1))
            if data.qpos[2]<.35 or tilt>np.pi/3:failed='SYNTHETIC_STANDING_FALL';break
        q=np.array(poses);path=out/f'seed_{seed}.npz';buf=io.BytesIO();np.savez(buf,qpos=q,reference=reference[:len(q)],timestamps=times[:len(q)],saturation=np.array(saturation));atomic_bytes(path,buf.getvalue())
        entry={'seed':seed,'status':'SYNTHETIC_STAND_PASS' if failed is None else 'FAIL','failure':failed,
               'completed_control_steps':len(q)-1,'required_control_steps':steps,'duration_s':float(times[len(q)-1]),
               'joint_rmse_rad':float(np.sqrt(np.mean((q[:,7:]-neutral[7:])**2))),
               'root_translation_rmse_m':float(np.sqrt(np.mean(np.sum((q[:,:3]-neutral[:3])**2,axis=1)))),
               'root_height_min_m':float(q[:,2].min()),'root_height_max_m':float(q[:,2].max()),
               'joint_speed_max_rad_s':max_speed,'torque_saturation_fraction':float(np.mean(saturation)),
               'trajectory_sha256':sha(path)}
        results.append(entry);print(entry,flush=True)
        write_json(ROOT/'audit/sonic_synthetic_stand_v1.json',{'status':'SYNTHETIC_CONTROLLER_INTEGRATION_ONLY',
            'scene':scene,'model_sha256':[r['sha256'] for r in interface.audit['models']],
            'initial_qpos':neutral.tolist(),'seed_noise_joint_std_rad':.002,'control_dt_s':dt,'sim_dt_s':model.opt.timestep,
            'standing_fall_rule':'root height <0.35 m or tilt >60 degrees; analytic stand only',
            'results':results,'dataset_clips_tested':0,'fusion_dynamic_gate_issued':False,
            'code_hashes':{str(p):sha(p) for p in [ROOT/'scripts/audit_sonic_stand.py',ROOT/'g1_pipeline/sonic_scene.py',ROOT/'g1_pipeline/sonic_observations.py']}})
    return results


if __name__=='__main__':run()
