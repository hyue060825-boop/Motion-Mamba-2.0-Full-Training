from pathlib import Path
import json, numpy as np, torch, sys
from scipy.spatial.transform import Rotation as R
from .common import ROOT,HML,write_json

def run_one():
 sys.path.insert(0,str(ROOT/'deps/GMR'));sys.path.insert(0,str(HML))
 from general_motion_retargeting import GeneralMotionRetargeting
 from human_body_prior.body_model.body_model import BodyModel
 import csv
 row=next(r for r in csv.DictReader(open(HML/'index.csv')) if r['new_name']=='000000.npy'); rel=Path(row['source_path'].removeprefix('./pose_data/')).with_suffix('.npz'); p=HML/'amass_data'/rel
 z=np.load(p); fps=float(z['mocap_framerate']);start=int(row['start_frame']);end=int(row['end_frame']);stride=max(1,int(fps/20));
 # apply official dataset preprocessing offset and crop exactly for this sample
 offset=0
 for name,val in [('Eyes_Japan_Dataset',60),('MPI_HDM05',60),('TotalCapture',20),('MPI_Limits',20),('Transitions_mocap',10)]:
  if name in str(rel):offset=val
 poses=z['poses'][offset:][start:end];trans=z['trans'][offset:][start:end];dmpls=z['dmpls'][offset:][start:end];gender=str(z['gender'].item());
 bm=BodyModel(bm_fname=str(HML/f'body_models/smplh/{gender}/model.npz'),dmpl_fname=str(HML/f'body_models/dmpls/{gender}/model.npz'),num_betas=16,num_dmpls=8)
 n=len(poses);device='cpu';
 with torch.no_grad():o=bm(root_orient=torch.tensor(poses[:, :3]).float(),pose_body=torch.tensor(poses[:,3:66]).float(),pose_hand=torch.tensor(poses[:,66:]).float(),trans=torch.tensor(trans).float(),betas=torch.tensor(np.repeat(z['betas'][:16][None],n,0)).float(),dmpls=torch.tensor(dmpls).float())
 j=o.Jtr.numpy();C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
 local=R.from_rotvec(o.full_pose.detach().cpu().numpy().reshape(n,-1,3)).as_matrix();parents=np.load(HML/f'body_models/smplh/{gender}/model.npz')['kintree_table'][0].astype(np.int64);parents[0]=-1
 globalrot=np.zeros_like(local)
 for k in range(local.shape[1]):globalrot[:,k]=local[:,k] if parents[k]<0 else np.einsum('fij,fjk->fik',globalrot[:,parents[k]],local[:,k])
 ids={'Hips':0,'Spine2':9,'LeftUpLeg':1,'LeftLeg':4,'LeftFootMod':7,'RightUpLeg':2,'RightLeg':5,'RightFootMod':8,'LeftArm':18,'LeftForeArm':20,'LeftHand':22,'RightArm':19,'RightForeArm':21,'RightHand':23}
 frames=[]
 for f in range(n):frames.append({name:[C@j[f,k],R.from_matrix(C@globalrot[f,k]@C.T).as_quat(scalar_first=True)] for name,k in ids.items()})
 # scale source meters and coordinates; Jtr is meters
 for d in frames:
  for k in d:d[k][0]=d[k][0]
 g=GeneralMotionRetargeting('bvh_lafan1','unitree_g1',actual_human_height=1.7,verbose=False,solver='daqp');stride2=max(1,int(round(fps/25)));q=[]
 for d in frames[::stride2]:q.append(g.retarget(d,offset_to_ground=False))
 a=np.asarray(q);res={'sample_id':'humanml3d:000000','native_path':str(p),'native_fps':fps,'frames':len(a),'qpos_shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'task_error_max':float(max(g.error1(),g.error2())),'status':'PASS' if a.shape[1]==36 and np.isfinite(a).all() else 'FAIL','adapter':'native SMPL-H FK to GMR human-frame landmarks; full local-pose global orientation transfer'}
 write_json(ROOT/'qa/aggregate/gmr_hml_one.json',res);return res
if __name__=='__main__':print(run_one())
