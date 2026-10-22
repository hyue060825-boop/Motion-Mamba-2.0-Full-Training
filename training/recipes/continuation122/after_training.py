"""Read-only queued acceptance; never starts or changes training."""
from motion_mamba2.paths import artifact_path
from common import *
import os,fcntl,subprocess,signal,traceback
from motion_training.io import rows,Budget
lock=(ROOT/'after_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
write(ROOT/'FINAL_REVIEW_CONTROLLER.json',dict(pid=os.getpid(),cwd=str(ROOT),command=__file__,time=time.time(),role='queued_evaluation_only_not_duplicate_training'))
cfg=read(ROOT/'config.json')
def state(s,**kw):write(ROOT/'FINAL_REVIEW_STATUS.json',dict(stage=s,time=time.time(),**kw))
def remaining(devices=1):
 start=read(ROOT/'RUN_START.json')['time'];used=sum(x['seconds']*x.get('devices',1) for x in rows(ROOT/'budget/GPU_TIME.jsonl'))
 for x in read(ROOT/'budget/ACTIVE.json').values():used+=(time.time()-x['start'])*x.get('devices',1)
 return min(cfg['wall_hours']*3600-(time.time()-start),(cfg['gpu_hours']*3600-used)/devices)-30

def run(name,script,args=(),devices=1):
 if (ROOT/f'{name}_COMPLETE.json').exists():return
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  state('WAITING_RESOURCES_FOR_ACCEPTANCE',next=name);assert remaining(devices)>0;time.sleep(30)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 timeout=remaining(devices);assert timeout>0
 with (ROOT/'logs'/(name+'.log')).open('a') as log:
  p=subprocess.Popen([PY,'-u',str(script)]+list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  state(name,pid=p.pid);write(ROOT/'FINAL_REVIEW_CHILD.json',dict(pid=p.pid,process_group=p.pid,script=str(script),args=list(map(str,args)),time=time.time()))
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM);p.wait();raise RuntimeError('Acceptance saved at wall/GPU budget boundary; no automatic extension')
  assert rc==0,(name,rc)
 write(ROOT/f'{name}_COMPLETE.json',dict(time=time.time(),rc=rc))
try:
 state('QUEUED_AFTER_TRAINING_AND_NATIVE_REVIEWS')
 while True:
  s=read(ROOT/'STATUS.json')
  if s['stage']=='COMPLETE_ADDITIONAL5000_ACTUAL122000':break
  if s['stage'] in ['FAILED_NEEDS_EXECUTION_REPAIR','SAVED_REPEATED_SEVERE_REGRESSION']:raise RuntimeError('Training did not complete: '+str(s))
  assert remaining(1)>0,'No wall budget remains'
  time.sleep(30)
 assert read(ROOT/'LATEST.json')['global_step']==122000
 run('parent117k_foundation_final','evaluate.py',['--checkpoint',cfg['parent'],'--step',5000,'--final'])
 run('parent117k_future_final','evaluate_future.py',['--checkpoint',cfg['parent'],'--step',5000,'--split','FINAL'])
 run('prepare_controlled_evaluation','setup_evaluation.py',devices=1)
 # Resource-use accounting is conservative: charge all8cards for this whole
 # acceptance subprocess, including CPU aggregation, never silently extend it.
 with Budget(cfg['budget_root']).timed('fixed506_controlled_evaluation_and_local52_quality',devices=8,closing=True):
  run('fixed506_controlled_evaluation',Path(artifact_path('controller_py_54c4155977')),devices=8)
 E=Path(artifact_path('native122k_controlled_endpoint_20261018_9d1c42d4fe'))
 assert (E/'COMPARISON_COMPLETE.json').exists()
 (ROOT/'FINAL_REPORT.md').write_text((E/'REPORT.md').read_text())
 write(ROOT/'FINAL_VERIFICATION.json',dict(status='COMPLETE5000_AND_PAIRED117K122K_ACCEPTANCE',parent=cfg['parent'],candidate=read(ROOT/'LATEST.json')['checkpoint'],actual_start=117000,actual_end=122000,updates=5000,
 evaluation=str(E),metrics='endpointCE; evaluator_v2 LocalFID52/R@3 per window; original foundation/interaction/physical checks; noRootADE',quality_improvement_not_assumed=True,time=time.time()))
 state('COMPLETE_TRAINING_AND_PAIRED_ACCEPTANCE')
except Exception:
 write(ROOT/'FINAL_REVIEW_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));state('FAILED_ACCEPTANCE_NEEDS_EXECUTION_REPAIR');raise
