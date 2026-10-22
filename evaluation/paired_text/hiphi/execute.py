"""New detailed-text requests; the admitted model loaders and physical sampler stay unchanged."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,importlib.util,argparse
R=Path(__file__).resolve().parent;P=Path(artifact_path('unified_interactive_easy_v3_hiphi_20261021_32d9db1045'))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=module('paired_v4_frozen_sampler',P/'execute.py');p.R=R
MODELS=['ardy','dip','dart','fd2','omg','cmdm','motionstreamer','native100','native110','native117','native122','minimal','motion_mamba2_official','motionmamba'];p.MODELS=MODELS
rd,wr,sha,digest=p.rd,p.wr,p.sha,p.digest

def schedule(task,case=None):
 phases=rd(R/'PLAN.json')['paired_phases'][task]
 if case is None:return ['forward']*len(phases),['']*len(phases)
 return [case['paired_instruction_root_states'][i] for i in phases],[case['paired_instruction_captions'][i] for i in phases]
p.schedule=schedule

def verify():
 f=rd(R/'FROZEN.json');assert f['passed']
 for n,h in f['hashes'].items():assert sha(R/n)==h,('changed_v4_input_or_source',n)
 for n,h in f['parent_sources'].items():assert sha(P/n)==h,('changed_parent_sampler',n)
 registry=rd(R/'CHECKPOINTS.json');base=rd(R/'BASE_CHECKPOINTS.json')
 for m in p.MODELS:
  if m not in registry:continue
  info=registry[m]
  if m!='motionmamba':assert info==base[m],('changed_baseline_identity',m)
  else:
   proof=rd(Path(info['training_root'])/'FINAL_SAVE_VERIFIED.json');assert proof['passed'] and proof['actual_step']==100000 and proof['checkpoint_sha256']==info['sha256']
  st=Path(info['checkpoint']).stat();f=info['frozen_file_stat'];assert st.st_size==f['size'] and st.st_mtime_ns==f['mtime_ns']
p.verify=verify
oldsetup=p.setup

def setup(method,out):
 if method!='motionmamba':return oldsetup(method,out)
 loader_path=Path(artifact_path('execute_py_308d54e9f2'))
 sys.path.insert(0,str(loader_path.parent))
 m=module('paired_v4_motionmamba_loader',loader_path);m.R=R
 return m.setup(method,out)
p.setup=setup
oldrun=p.run

def run(method,model,m,client,case,task,seed,assets,features,gate=False,timing=False):
 old=p.schedule;p.schedule=lambda task:schedule(task,case)
 try:geo,audit=oldrun(method,model,m,client,case,task,seed,assets,features,gate=gate,timing=timing)
 finally:p.schedule=old
 audit['paired_caption_ids']=[case['paired_instruction_records'][i]['sample_id'] for i in rd(R/'PLAN.json')['paired_phases'][task]]
 audit['detailed_text_pairing']=True;audit['no_paired_GT_motion_supplied_to_generator']=True
 return geo,audit
p.run=run

def worker(method,task):
 assert method in rd(R/'CHECKPOINTS.json'),'Requested checkpoint not complete and bound'
 p.MODELS=[method];verify();return p.worker(method,task)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',choices=MODELS,required=True);a.add_argument('--task',choices=['short52','nominal9','nominal20','nominal40'],required=True);v=a.parse_args()
 try:worker(v.model,v.task)
 except BaseException:
  import traceback,time;wr(R/'models'/v.model/v.task/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
