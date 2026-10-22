from pathlib import Path
import json,numpy as np,mujoco
from .common import ROOT,write_json

def run():
 m=mujoco.MjModel.from_xml_path(str(ROOT/'deps/GMR/assets/unitree_g1/g1_mocap_29dof.xml'));d=mujoco.MjData(m);rows=[]
 floor_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
 floor_pen= self_pen=0; root_z=[]
 for p in sorted((ROOT/'work/hml_smoke_qpos').glob('*.npy')):
  q=np.load(p);bad=0;penetration=0
  for x in q:
   if x.shape!=(36,) or not np.isfinite(x).all():bad+=1;continue
   d.qpos[:]=x;mujoco.mj_forward(m,d); root_z.append(float(x[2]))
   if not np.isfinite(d.xpos).all() or not np.isfinite(d.qvel).all():bad+=1
   floor_bad=False; self_bad=False
   for ci in range(d.ncon):
    c=d.contact[ci]; bad=bool(c.dist < -0.05)
    if bad:
     if c.geom1==floor_id or c.geom2==floor_id: floor_bad=True
     else: self_bad=True
   floor_pen += int(floor_bad); self_pen += int(self_bad)
   penetration += int(floor_bad or self_bad)
  rows.append({'file':str(p.relative_to(ROOT)),'frames':len(q),'invalid':bad,'deep_penetration_frames':penetration,'status':'PASS' if bad==0 and penetration==0 else ('KINEMATIC_NUMERIC_PASS_GEOMETRY_FAIL' if bad==0 else 'FAIL')})
 write_json(ROOT/'qa/aggregate/mujoco_hml_kinematic.json',{'clips':len(rows),'frames':sum(x['frames'] for x in rows),'invalid_frames':sum(x['invalid'] for x in rows),'penetration_frames':sum(x['deep_penetration_frames'] for x in rows),'floor_penetration_frames':floor_pen,'self_penetration_frames':self_pen,'root_z_percentiles_m':np.percentile(root_z,[1,5,50,95,99]).tolist() if root_z else None,'numeric_status':'PASS' if rows and all(x['invalid']==0 for x in rows) else 'FAIL','geometry_status':'PASS' if rows and all(x['status']=='PASS' for x in rows) else 'FAIL','model':'GMR unitree_g1/mocap, not locked ARDY model'});return rows
if __name__=='__main__':print(run()[-1])
