"""Full-duration SONIC tracking on the exact locked actuated G1 scene.

The caller must bind a protocol before evaluation; this runner never changes
thresholds or reference motion and cannot run before both source smoke gates.
"""
import io
import json
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,sha,atomic_bytes,write_json
from .gates import require_pass
from .sonic_scene import prepare_scene,torque_control
from .sonic_observations import SonicObservations
from .resample import qpos as resample


class Tracker:
    def __init__(self,protocol_path):
        require_pass('03_hiphi100')
        self.path=protocol_path;self.protocol=json.loads(protocol_path.read_text())
        if self.protocol['status']!='FROZEN_BEFORE_DYNAMIC_PANEL':raise ValueError('DYNAMIC_PROTOCOL_NOT_FROZEN')
        for path,bound in self.protocol['bindings'].items():
            if sha(ROOT/path)!=bound:raise ValueError('DYNAMIC_DEPENDENCY_CHANGED:'+path)
        sys.path.insert(0,str(ROOT/'deps/onnxruntime_1_22_1'))
        import onnxruntime as ort
        if ort.__version__!='1.22.1':raise ValueError('DYNAMIC_ONNXRUNTIME_VERSION')
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        interface=SonicObservations();self.sessions=[]
        for entry in interface.audit['models']:
            if sha(entry['path'])!=entry['sha256']:raise ValueError('SONIC_MODEL_CHANGED')
            self.sessions.append(ort.InferenceSession(entry['path'],sess_options=options,providers=['CPUExecutionProvider']))
        self.model,self.scene=prepare_scene();self.dt=.02;self.substeps=round(self.dt/self.model.opt.timestep)
        if abs(self.substeps*self.model.opt.timestep-self.dt)>1e-10:raise ValueError('DYNAMIC_DT_NOT_INTEGER')

    def track(self,q25,t25,*,seed,scene_class,output):
        if seed not in self.protocol['seeds']:raise ValueError('UNFROZEN_DYNAMIC_SEED')
        if scene_class not in self.protocol['supported_scenes']:raise ValueError('UNSUPPORTED_DYNAMIC_SCENE')
        if not np.allclose(np.diff(t25),.04,atol=1e-6,rtol=0):raise ValueError('DYNAMIC_REFERENCE_NOT_25HZ')
        model=self.model;ref,times,mapping=resample(q25,t25,model.jnt_range[1:],fps=50.)
        if abs(times[-1]-t25[-1])>1e-6:raise ValueError('DYNAMIC_REFERENCE_DURATION_CHANGED')
        velocity=np.empty((len(ref),35))
        for f in range(len(ref)-1):mujoco.mj_differentiatePos(model,velocity[f],self.dt,ref[f],ref[f+1])
        velocity[-1]=velocity[-2]
        obs=SonicObservations();data=mujoco.MjData(model);data.qpos[:]=ref[0];data.qvel[:]=velocity[0]
        data.qpos[7:]+=np.random.default_rng(seed).normal(0,self.protocol['initial_joint_noise_std_rad'],29)
        mujoco.mj_forward(model,data);targetdata=mujoco.MjData(model)
        feet=[model.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')]
        poses=[data.qpos.copy()];velocities=[data.qvel.copy()];footerr=[];torques=[];saturation=[];collisions=[];last=np.zeros(29);failure=None;divergence=0
        thresholds=self.protocol['per_clip_limits'];control_contacts=[]
        for f in range(len(ref)-1):
            gyro=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,model.body('pelvis').id,gyro,1)
            obs.push_state(data.qpos[7:],data.qvel[6:],data.qpos[3:7],gyro[:3],last,time_s=float(times[f]))
            token=self.sessions[0].run(None,{'obs_dict':obs.encoder(ref[:,7:],velocity[:,6:],ref[:,3:7],f,data.qpos[3:7],timestamps=times)})[0]
            last=self.sessions[1].run(None,{'obs_dict':obs.decoder(token)})[0].reshape(29);target=obs.joint_target(last)
            forbidden_depth=0.;contacts=0
            for _ in range(self.substeps):
                torque,info=torque_control(model,data.qpos,data.qvel,target,obs.constants)
                data.ctrl[:]=torque;torques.append(torque.copy());saturation.append(info['saturated']);mujoco.mj_step(model,data)
                for contact in data.contact:
                    if contact.dist>=0:continue
                    b1,b2=model.geom_bodyid[contact.geom1],model.geom_bodyid[contact.geom2]
                    if b1 and b2:forbidden_depth=max(forbidden_depth,-float(contact.dist))
                    else:contacts+=1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():failure='NONFINITE_SIMULATION';break
            mujoco.mj_forward(model,data);targetdata.qpos[:]=ref[f+1];mujoco.mj_forward(model,targetdata)
            footerr.append(np.linalg.norm(data.xpos[feet]-targetdata.xpos[feet],axis=1))
            collisions.append(forbidden_depth);control_contacts.append(contacts);poses.append(data.qpos.copy());velocities.append(data.qvel.copy())
            orientation=float(np.rad2deg((R.from_quat(ref[f+1,3:7],scalar_first=True).inv()*R.from_quat(data.qpos[3:7],scalar_first=True)).magnitude()))
            lost_height=ref[f+1,2]-data.qpos[2]>thresholds['reference_relative_height_loss_max_m']
            diverged=orientation>thresholds['reference_relative_orientation_divergence_deg']
            divergence=divergence+1 if diverged else 0
            if lost_height or divergence*self.dt>=thresholds['orientation_divergence_duration_s']:
                failure='REFERENCE_RELATIVE_FALL_OR_DIVERGENCE';break
        actual=np.asarray(poses);r=ref[:len(actual)];actual_velocity=np.asarray(velocities);error=actual[:,7:]-r[:,7:]
        orientation=np.rad2deg((R.from_quat(r[:,3:7],scalar_first=True).inv()*R.from_quat(actual[:,3:7],scalar_first=True)).magnitude())
        metric={'joint_rmse_rad':float(np.sqrt(np.mean(error**2))),'per_joint_rmse_rad':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'root_translation_rmse_m':float(np.sqrt(np.mean(np.sum((actual[:,:3]-r[:,:3])**2,axis=1)))),
            'root_orientation_mean_deg':float(orientation.mean()),'root_orientation_p95_deg':float(np.percentile(orientation,95)),
            'feet_position_mean_m':np.asarray(footerr).mean(axis=0).tolist() if footerr else None,
            'torque_saturation_fraction':float(np.mean(saturation)) if saturation else None,
            'joint_speed_max_rad_s':float(abs(actual_velocity[:,6:]).max()),
            'per_joint_speed_max_rad_s':abs(actual_velocity[:,6:]).max(axis=0).tolist(),
            'self_collision_max_depth_m':max(collisions,default=0.),
            'self_collision_over_1cm_control_fraction':float(np.mean(np.asarray(collisions)>.01)) if collisions else None,
            'motor_velocity_saturation_fraction':None,'motor_velocity_saturation_reason':'No verified matched motor velocity envelope; actual speed is recorded without claiming hardware compliance',
            'full_duration_completed':failure is None and len(actual)==len(ref),'failure':failure,'seed':seed,'scene_class':scene_class,
            'expected_control_steps':len(ref)-1,'completed_control_steps':len(actual)-1,'reference_duration_s':float(times[-1]-times[0]),
            'successful_duration_s':float(times[len(actual)-1]-times[0]),'protocol_sha256':sha(self.path),'actuated_scene_sha256':self.scene['actuated_xml_sha256']}
        reasons=[]
        if not metric['full_duration_completed']:reasons.append(failure or 'INCOMPLETE_DURATION')
        for key in ['joint_rmse_rad','root_translation_rmse_m','root_orientation_mean_deg']:
            if metric[key]>thresholds[key]:reasons.append(key)
        if metric['feet_position_mean_m'] is None or max(metric['feet_position_mean_m'])>thresholds['feet_position_mean_m']:reasons.append('feet_position_mean_m')
        metric.update(status='PASS' if not reasons else 'FAIL',failure_reasons=reasons)
        buf=io.BytesIO();np.savez_compressed(buf,qpos=actual,reference=ref,timestamps=times,actual_qvel=actual_velocity,
            torques=np.asarray(torques),torque_saturation=np.asarray(saturation),self_collision_depth=np.asarray(collisions),ground_contact_counts=np.asarray(control_contacts))
        atomic_bytes(output,buf.getvalue());metric['trajectory_path']=str(output.relative_to(ROOT));metric['trajectory_sha256']=sha(output)
        write_json(output.with_suffix('.json'),metric);return metric
