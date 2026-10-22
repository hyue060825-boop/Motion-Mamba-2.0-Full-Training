from pathlib import Path
import json,numpy as np,torch
from .common import ROOT,write_json
from .robot import encode_csv
from motion_mamba2.exports.mujoco import MujocoQposConverter

def run():
 import sys;sys.path.insert(0,str(ROOT.parent/'x'))
 out=[]
 for p in sorted((ROOT/'work/hml_smoke_csv').glob('*.csv')):
  try:
   f,rep,_=encode_csv(p);inv=rep.inverse(f,is_normalized=False);q=MujocoQposConverter(rep.skeleton).dict_to_qpos(inv)[0];orig=None
   out.append({'file':str(p.relative_to(ROOT)),'frames':len(f[0]),'inverse_finite':bool(np.isfinite(q).all()),'status':'PASS' if np.isfinite(q).all() else 'FAIL'})
  except Exception as e:out.append({'file':str(p.relative_to(ROOT)),'status':'ERROR','error':repr(e)})
 write_json(ROOT/'qa/aggregate/hml_roundtrip.json',{'attempted':len(out),'passed':sum(x['status']=='PASS' for x in out),'errors':sum(x['status']=='ERROR' for x in out),'status':'PASS' if len(out)==100 and all(x['status']=='PASS' for x in out) else 'FAIL'});return out
if __name__=='__main__':print(run()[-1])
