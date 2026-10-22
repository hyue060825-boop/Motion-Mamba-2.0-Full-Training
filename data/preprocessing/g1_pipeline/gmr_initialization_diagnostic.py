"""Development-only first-frame initialization experiment; no reference pose input.

The bounded orientation seed follows the compiled robot chains from pelvis to
extremities. GMR remains the retargeter. This module never reads accept panels.
"""
import copy
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .common import ROOT, read_jsonl, write_json, sha
from .locked_target import prepare, create_retargeter, initialize_root_from_human
from .gmr_soma import parse


CHAINS=[('torso_link',['waist_yaw','waist_roll','waist_pitch'])]
for side in ('left','right'):
    CHAINS.extend([(f'{side}_hip_yaw_link',[f'{side}_hip_{axis}' for axis in ('pitch','roll','yaw')]),
                   (f'{side}_knee_link',[f'{side}_knee']),
                   (f'{side}_ankle_roll_link',[f'{side}_ankle_pitch',f'{side}_ankle_roll']),
                   (f'{side}_shoulder_yaw_link',[f'{side}_shoulder_{axis}' for axis in ('pitch','roll','yaw')]),
                   (f'{side}_elbow_link',[f'{side}_elbow']),
                   (f'{side}_wrist_yaw_link',[f'{side}_wrist_{axis}' for axis in ('roll','pitch','yaw')])])


def orientation_seed(solver,human_frame):
    initialize_root_from_human(solver,copy.deepcopy(human_frame))
    model=solver.model;data=mujoco.MjData(model);q=solver.configuration.q.copy()
    for body,joints in CHAINS:
        ids=[model.joint(n+'_joint').id for n in joints];adr=model.jnt_qposadr[ids]
        lower=model.jnt_range[ids,0]+1e-7;upper=model.jnt_range[ids,1]-1e-7
        target=R.from_quat(solver.scaled_human_data[solver.ik_match_table1[body][0]][1],scalar_first=True)
        bid=model.body(body).id
        def residual(x):
            data.qpos[:]=q;data.qpos[adr]=x;mujoco.mj_kinematics(model,data)
            return (target.inv()*R.from_matrix(data.xmat[bid].reshape(3,3))).as_rotvec()
        starts=[np.clip(q[adr],lower,upper),(lower+upper)/2,lower*.8+upper*.2,lower*.2+upper*.8]
        fits=[least_squares(residual,start,bounds=(lower,upper),max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8) for start in starts]
        best=min(fits,key=lambda f:(round(float(np.dot(f.fun,f.fun)),8),float(np.linalg.norm(f.x))))
        q[adr]=best.x
    solver.configuration.update(q)
    return q.copy()


def task_cost(solver,stage=2):
    return float(sum(np.sum((t.cost*t.compute_error(solver.configuration))**2) for t in getattr(solver,f'tasks{stage}')))


def run(max_clips=200):
    namespace='bones_soma_rootweight_dev_v6'
    config_path=ROOT/'configs/bones_soma_rootweight_v6.json';cfg=json.loads(config_path.read_text())
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';xml,_=prepare();out=[]
    for row in read_jsonl(manifest)[:max_clips]:
        mid=row['source_motion_id'];bvh=ROOT/'work/bones_pairs_dev'/(mid+'.bvh')
        frames,_=parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=np.array([0.]),rotation_basis='world_only')
        solver=create_retargeter(config_path,xml);qbase=np.load(ROOT/'work'/namespace/(mid+'.npy'))[0]
        solver.update_targets(copy.deepcopy(frames[0]));solver.configuration.update(qbase);basecost=task_cost(solver)
        seed=orientation_seed(solver,frames[0]);q=solver.retarget(copy.deepcopy(frames[0]),offset_to_ground=False).copy()
        rowout={'sample_id':row['sample_id'],'baseline_cost':basecost,'seeded_cost':task_cost(solver),
                'orientation_seed_qpos':seed.tolist(),'seeded_gmr_qpos':q.tolist(),'source_sha256':sha(bvh)}
        out.append(rowout);print(row['sample_id'],basecost,rowout['seeded_cost'],flush=True)
        write_json(ROOT/'audit/gmr_initialization_dev_diagnostic.json',{'status':'DEVELOPMENT_DIAGNOSTIC','manifest_sha256':sha(manifest),
            'config_sha256':sha(config_path),'code_sha256':sha(__file__),'selection':'fixed dev manifest order; no reference angles used',
            'metric':'stage 2 sum squared Mink cost-weighted task errors','results':out})
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--max-clips',type=int,default=200);a=p.parse_args();run(a.max_clips)
