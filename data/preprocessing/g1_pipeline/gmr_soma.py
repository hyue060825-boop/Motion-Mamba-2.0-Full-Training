from __future__ import annotations
import json,tarfile,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import zstandard as zstd
from .common import ROOT,BONES,write_json,write_jsonl

MAP={'Hips':'Hips','Spine2':'Spine2','LeftLeg':'LeftUpLeg','LeftShin':'LeftLeg','LeftToeBase':'LeftFootMod','RightLeg':'RightUpLeg','RightShin':'RightLeg','RightToeBase':'RightFootMod','LeftArm':'LeftArm','LeftForeArm':'LeftForeArm','LeftHand':'LeftHand','RightArm':'RightArm','RightForeArm':'RightForeArm','RightHand':'RightHand'}
HIPHI_MAP={'Hips':'Hips','Spine2':'Spine2','LeftUpLeg':'LeftUpLeg','LeftLeg':'LeftLeg','LeftFoot':'LeftFootMod','RightUpLeg':'RightUpLeg','RightLeg':'RightLeg','RightFoot':'RightFootMod','LeftArm':'LeftArm','LeftForeArm':'LeftForeArm','LeftHand':'LeftHand','RightArm':'RightArm','RightForeArm':'RightForeArm','RightHand':'RightHand'}

def parse(path, mapping=MAP, *, target_timestamps=None, rotation_basis='conjugate'):
 lines=Path(path).read_text().splitlines(); joints=[];stack=[];channels=[];i=0
 while i<len(lines):
  s=lines[i].strip(); p=s.split()
  if s=='MOTION':break
  if p and p[0] in ('ROOT','JOINT'):
   parent=stack[-1] if stack else None;joints.append({'name':p[1],'parent':parent,'offset':None,'channels':[]});stack.append(len(joints)-1)
  elif p and p[0]=='End' and len(p)>1:stack.append(None)
  elif p and p[0]=='OFFSET' and stack and stack[-1] is not None:joints[stack[-1]]['offset']=np.array([float(x) for x in p[1:]])
  elif p and p[0]=='CHANNELS' and stack and stack[-1] is not None:
   joints[stack[-1]]['channels']=p[2:2+int(p[1])];channels.extend((stack[-1],x) for x in joints[stack[-1]]['channels'])
  elif s=='}' and stack:stack.pop()
  i+=1
 n=int(lines[i+1].split(':')[1]);dt=float(lines[i+2].split(':')[1]); vals=np.asarray([[float(x) for x in lines[i+3+k].split()] for k in range(n)])
 jidx={j['name']:k for k,j in enumerate(joints)}; ch={k:[] for k in range(len(joints))}
 for ci,(ji,c) in enumerate(channels):ch[ji].append((ci,c))
 sampling=None
 if target_timestamps is not None:
  from .resample import time_grid
  target_timestamps=np.asarray(target_timestamps,dtype=float)
  out,sampling=time_grid(np.arange(n)*dt, start=float(target_timestamps[0]),end=np.nextafter(float(target_timestamps[-1]),np.inf))
  if out.shape!=target_timestamps.shape or not np.allclose(out,target_timestamps,atol=1e-10,rtol=0): raise ValueError('BVH_TARGET_TIME_MISMATCH')
 out_n=n if sampling is None else len(target_timestamps)
 wp=np.zeros((out_n,len(joints),3));wr=np.zeros((out_n,len(joints),3,3))
 for k,j in enumerate(joints):
  pos=np.tile(j['offset'],(n,1)); seq=''; inds=[]
  for ci,c in ch[k]:
   if c.endswith('position'):pos[:, 'XYZ'.index(c[0])]=vals[:,ci]
   elif c.endswith('rotation'):seq+=c[0].lower();inds.append(ci)
  rot=R.from_euler(seq.upper(),vals[:,inds],degrees=True).as_matrix() if seq else np.tile(np.eye(3),(n,1,1))
  if sampling is not None:
   from .resample import linear,rotations
   pos=linear(pos,sampling);rot=rotations(rot,sampling)
  if j['parent'] is None:wp[:,k]=pos;wr[:,k]=rot
  else:
   pp=j['parent'];wp[:,k]=wp[:,pp]+np.einsum('fij,fj->fi',wr[:,pp],pos);wr[:,k]=np.einsum('fij,fjk->fik',wr[:,pp],rot)
 # BVH cm Y-up -> GMR m Z-up/X-forward basis, same as GMR lafan loader
 C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
 frames=[]
 if rotation_basis not in ('conjugate','world_only'): raise ValueError('UNKNOWN_ROTATION_BASIS')
 for f in range(out_n):
  d={}
  for src,dst in mapping.items():
   if src not in jidx: raise ValueError('MISSING_SOMA_JOINT_'+src)
   k=jidx[src]; mat=C@wr[f,k]; pos=(wp[f,k]@C.T)/100
   if rotation_basis=='conjugate': mat=mat@C.T
   d[dst]=[pos,R.from_matrix(mat).as_quat(scalar_first=True)]
  frames.append(d)
 return frames,1/dt

def extract_one(rel,out):
 import csv
 meta=next(r for r in csv.DictReader(open(BONES/'metadata/seed_metadata_v004.csv')) if r['filename']==rel)
 tarpath=BONES/'soma_uniform.tar.gz'; member=meta['move_soma_uniform_path']
 with tarfile.open(tarpath,'r|gz') as tar:
  for m in tar:
   if m.name==member:
    b=tar.extractfile(m).read();p=Path(out);p.write_bytes(b);return p
 raise FileNotFoundError(member)

def extract_many(rows,out):
 import csv
 wanted={r['source_motion_id']:next(x for x in csv.DictReader(open(BONES/'metadata/seed_metadata_v004.csv')) if x['filename']==r['source_motion_id']) for r in rows}
 members={x['move_soma_uniform_path']:mid for mid,x in wanted.items()};found={}
 for mid in wanted:
  p=Path(out)/(mid+'.bvh')
  if p.is_file(): found[mid]=p
 if len(found)==len(members): return found
 with tarfile.open(BONES/'soma_uniform.tar.gz','r|gz') as tar:
  for m in tar:
   mid=members.get(m.name)
   if mid:
    p=Path(out)/(mid+'.bvh');p.write_bytes(tar.extractfile(m).read());found[mid]=p
    if len(found)==len(members):break
 return found

def run_one():
 import sys
 sys.path.insert(0,str(ROOT/'deps/GMR'))
 from general_motion_retargeting import GeneralMotionRetargeting
 rows=[json.loads(x) for x in (ROOT/'manifests/bones_pairs_smoke.jsonl').read_text().splitlines()]
 row=rows[0]; rel=row['source_motion_id'];
 work=ROOT/'work/gmr_soma';work.mkdir(parents=True,exist_ok=True);bvh=extract_one(rel,work/(rel+'.bvh'))
 frames,fps=parse(bvh); g=GeneralMotionRetargeting('bvh_lafan1','unitree_g1',actual_human_height=row['height_m'],verbose=False,solver='daqp',use_velocity_limit=False)
 stride=max(1,int(round(fps/25))); selected=frames[::stride][:max(2,int((len(frames)-1)/stride)+1)]
 q=[]; errs=[]
 for d in selected:
  try:q.append(g.retarget(d,offset_to_ground=False));errs.append({'task1':float(g.error1()),'task2':float(g.error2())})
  except Exception as e: raise RuntimeError(f'GMR frame failed: {e}')
 q=np.asarray(q)
 result={'sample_id':row['sample_id'],'source':str(bvh),'source_fps':fps,'target_frames':len(q),'qpos_shape':list(q.shape),
         'finite':bool(np.isfinite(q).all()),'qpos_width':q.shape[1],'task_error_max':max(max(x.values()) for x in errs),
         'target_robot_xml':str(g.xml_file),'solver':'daqp','status':'PASS' if q.shape[1]==36 and np.isfinite(q).all() else 'FAIL'}
 write_json(ROOT/'qa/aggregate/gmr_soma_one.json',result);return result

def run_smoke():
 import sys
 sys.path.insert(0,str(ROOT/'deps/GMR'))
 from general_motion_retargeting import GeneralMotionRetargeting
 rows=[json.loads(x) for x in (ROOT/'manifests/bones_pairs_smoke.jsonl').read_text().splitlines()]
 work=ROOT/'work/gmr_soma_smoke';work.mkdir(parents=True,exist_ok=True);paths=extract_many(rows,work)
 g=GeneralMotionRetargeting('bvh_lafan1','unitree_g1',verbose=False,solver='daqp',use_velocity_limit=False);results=[]
 for row in rows:
  try:
   frames,fps=parse(paths[row['source_motion_id']]);stride=max(1,int(round(fps/25)));sel=frames[::stride]
   q=[];errs=[]
   for d in sel:q.append(g.retarget(d,offset_to_ground=False));errs.append(max(float(g.error1()),float(g.error2())))
   a=np.asarray(q);np.save(work/(row['source_motion_id']+'.qpos.npy'),a)
   # Diagnostic only: GMR and BONES/ARDY target assets are not yet proven identical.
   from .robot import read_csv
   refpath=BONES/next(x for x in __import__('csv').DictReader(open(BONES/'metadata/seed_metadata_v004.csv')) if x['filename']==row['source_motion_id'])['move_g1_path']
   ref,rt,rfps=read_csv(refpath,format_id='bones_g1_frame_cm_euler_xyz_deg_hinge_deg_v1',fps=120)
   from .resample import qpos as rs
   ref,_,_=rs(ref,rt,np.asarray(__import__('mujoco').MjModel.from_xml_path(str(ROOT/'deps/GMR/assets/unitree_g1/g1_mocap_29dof.xml')).jnt_range[1:]))
   n=min(len(a),len(ref));
   results.append({'sample_id':row['sample_id'],'frames':len(a),'qpos_shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'task_error_max':max(errs),'reference_frames':len(ref),'diagnostic_root_rmse_m':float(np.sqrt(np.mean((a[:n,:3]-ref[:n,:3])**2))),'diagnostic_hinge_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((a[:n,7:]-ref[:n,7:])**2)))),'asset_comparison':'DIAGNOSTIC_ONLY_UNVERIFIED','status':'PASS' if a.shape[1]==36 and np.isfinite(a).all() else 'FAIL'})
  except Exception as e:results.append({'sample_id':row['sample_id'],'status':'ERROR','error':repr(e)})
 write_jsonl(ROOT/'manifests/bones_gmr_smoke_results.jsonl',results);write_json(ROOT/'qa/aggregate/gmr_soma_smoke.json',{'attempted':len(rows),'results':results,'passed':sum(x['status']=='PASS' for x in results),'status':'PASS' if all(x['status']=='PASS' for x in results) else 'FAIL','target_asset':str(g.xml_file),'asset_semantics':'GMR unitree_g1 g1_mocap_29dof; reference BONES/ARDY asset comparison not yet accepted'})
 return results

if __name__=='__main__':print(run_smoke())

def run_hiphi_one():
 import sys,shutil
 sys.path.insert(0,str(ROOT/'deps/GMR'))
 from general_motion_retargeting import GeneralMotionRetargeting
 from general_motion_retargeting import params
 cfg=Path(ROOT/'deps/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json'); custom=ROOT/'configs/bvh_hiphi_to_g1.json';
 if not custom.exists():shutil.copy(cfg,custom)
 params.IK_CONFIG_DICT['bvh_hiphi']={'unitree_g1':custom}
 files=list((ROOT/'work/hiphi_packages').glob('*.bvh'))
 if not files: raise FileNotFoundError('No extracted HiPHI smoke BVH')
 path=files[0];frames,fps=parse(path,HIPHI_MAP);g=GeneralMotionRetargeting('bvh_hiphi','unitree_g1',verbose=False,solver='daqp');stride=max(1,int(round(fps/25)));q=[]
 for d in frames[::stride]:q.append(g.retarget(d,offset_to_ground=False))
 a=np.asarray(q);res={'path':str(path),'source_fps':fps,'frames':len(a),'qpos_shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'task_error_max':float(max(g.error1(),g.error2())),'status':'PASS' if a.shape[1]==36 and np.isfinite(a).all() else 'FAIL','adapter':'bvh_hiphi custom semantic name mapping; IK weights inherited only as development baseline'}
 write_json(ROOT/'qa/aggregate/gmr_hiphi_one.json',res);return res
