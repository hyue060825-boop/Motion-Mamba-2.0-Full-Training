"""Measure constant body-frame corrections on the existing smoke development panel.
No held-out acceptance samples are inspected and no release config is fitted here.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,REPO,read_jsonl,write_json,sha,setup_imports
from .robot import compile_target,read_csv,FORMAT,BASIS
from .gmr_soma import parse
from .resample import qpos as resample_qpos

PAIRS={'pelvis':'Hips','torso_link':'Chest',
       'left_hip_yaw_link':'LeftLeg','left_knee_link':'LeftShin','left_ankle_roll_link':'LeftFoot',
       'right_hip_yaw_link':'RightLeg','right_knee_link':'RightShin','right_ankle_roll_link':'RightFoot',
       'left_shoulder_yaw_link':'LeftArm','left_elbow_link':'LeftForeArm','left_wrist_yaw_link':'LeftHand',
       'right_shoulder_yaw_link':'RightArm','right_elbow_link':'RightForeArm','right_wrist_yaw_link':'RightHand'}

def run():
    setup_imports();import torch
    from motion_mamba2.skeleton import SOMASkeleton77
    skel=SOMASkeleton77(load=False)
    rest_path=REPO/'ardy/ardy/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p'
    rest=torch.load(rest_path,map_location='cpu',weights_only=True).squeeze().numpy()
    model,_=compile_target();data=mujoco.MjData(model)
    c=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    # BONES world axes follow the BVH loader, not ARDY representation axes.
    basis=c; basis_change=basis@c.T
    matrices={k:[] for k in PAIRS};clips=[];source_planar=[];reference_planar=[]
    from .common import BONES
    for row in read_jsonl(ROOT/'manifests/bones_pairs_smoke.jsonl'):
        bvh=ROOT/'work/gmr_soma_smoke'/(row['source_motion_id']+'.bvh')
        timing=__import__('json').loads((ROOT/'work/bones_locked_smoke_v1'/(row['source_motion_id']+'.time.json')).read_text())
        times=np.asarray(timing['source']['target_timestamps'])
        human,_=parse(bvh,{n:n for n in PAIRS.values()},target_timestamps=times,rotation_basis='world_only')
        q,t,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
        q,_,_=resample_qpos(q,t,model.jnt_range[1:],end=np.nextafter(times[-1],np.inf))
        hp=np.asarray([f['Hips'][0] for f in human])
        source_planar.extend(hp[:,:2]-hp[:1,:2]);reference_planar.extend(q[:,:2]-q[:1,:2])
        clip={k:[] for k in PAIRS}
        for frame,x in zip(human,q):
            data.qpos[:]=x;mujoco.mj_forward(model,data)
            for body,source in PAIRS.items():
                sr=basis_change@R.from_quat(frame[source][1],scalar_first=True).as_matrix()
                tr=data.xmat[model.body(body).id].reshape(3,3)
                clip[body].append(sr.T@tr)
        for body in PAIRS:matrices[body].extend(clip[body])
        clips.append({'sample_id':row['sample_id'],'frames':len(q)})
    result={}
    for body,values in matrices.items():
        rot=R.from_matrix(np.asarray(values));mean=rot.mean();err=np.rad2deg((mean.inv()*rot).magnitude())
        source=PAIRS[body];raw_rest=basis@rest[skel.bone_order_names.index(source)]
        result[body]={'source':source,'constant_offset_wxyz':mean.as_quat(scalar_first=True).tolist(),
            'residual_mean_deg':float(err.mean()),'residual_p95_deg':float(np.percentile(err,95)),
            'neutral_body_frame_after_offset':(raw_rest@mean.as_matrix()).tolist()}
    sp=np.asarray(source_planar);rp=np.asarray(reference_planar)
    correlation=np.sum((rp[:,0]+1j*rp[:,1])*(sp[:,0]-1j*sp[:,1]))
    report={'scope':'smoke panel diagnostic of constant offsets, not an independently validated config',
        'trajectory_diagnostic':{'extra_world_yaw_deg':float(np.rad2deg(np.angle(correlation))),
                                 'uniform_planar_scale':float(abs(correlation)/np.sum(sp*sp))},
        'basis_soma_to_robot':basis.tolist(),'source_rest_sha256':sha(rest_path),'clips':clips,'body_offsets':result}
    write_json(ROOT/'qa/aggregate/soma_constant_offset_diagnostic.json',report)
    print({k:round(v['residual_mean_deg'],2) for k,v in result.items()})
if __name__=='__main__':run()
