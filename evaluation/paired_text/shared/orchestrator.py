"""Exclusive finite HiPHI evaluation pipeline; never trains or changes live models."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,time,json,fcntl,subprocess,traceback,hashlib
R=Path(__file__).resolve().parent
PY=artifact_path('python_9f8003379b')
def wr(n,x):
 p=R/n;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
 lock=(R/'orchestrator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wr('ORCHESTRATOR_PID.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),script=str(Path(__file__).resolve()),time=time.time()))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 for script,receipt in [('build_protocol.py','FROZEN.json'),('prepare_scoring.py','scoring/REFERENCE_COMPLETE.json')]:
  if (R/receipt).exists():continue
  with (R/(script+'.log')).open('a') as log:
   p=subprocess.Popen([PY,str(R/script)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   wr('PREPARATION_PID.json',dict(pid=p.pid,script=script,cwd=str(R),CPU_only=True,time=time.time()));wr('PIPELINE_STATUS.json',dict(stage='CPU_PREPARATION',script=script,pid=p.pid,time=time.time()))
   assert p.wait()==0,(script,'failed; inspect original log')
  assert (R/receipt).exists(),('missing_preparation_receipt',receipt)
 wr('EXECUTION_SOURCE_MANIFEST.json',dict(hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},time=time.time()))
 children={}
 for script in ['controller.py','scoring_controller.py','retiming_controller.py']:
  # Exclusive child locks also prevent accidental duplication during recovery.
  log=(R/(script+'.log')).open('a')
  p=subprocess.Popen([PY,str(R/script)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  children[script]=dict(p=p,log=log)
 wr('PIPELINE_CHILDREN.json',dict(workers=[dict(script=n,pid=v['p'].pid) for n,v in children.items()],time=time.time()))
 while children:
  for n,v in list(children.items()):
   rc=v['p'].poll()
   if rc is None:continue
   v['log'].close();del children[n];assert rc==0,(n,rc)
  for f in ['FAILURE.json','scoring/FAILURE.json','retiming/FAILURE.json']:
   if (R/f).exists():raise RuntimeError('Execution failure recorded in '+f+'; preserve completed work and repair only execution')
  wr('PIPELINE_STATUS.json',dict(stage='GENERATING_GATING_SCORING_AND_TIMING',active=[dict(script=n,pid=v['p'].pid) for n,v in children.items()],time=time.time()))
  if children:time.sleep(15)
 complete=json.loads((R/'delivery/COMPLETE.json').read_text());assert complete['passed']
 wr('REMOTE_COMPLETE.json',dict(passed=True,checkpoints=13,conditions=52,rollouts=19968,delivery=str(R/'delivery'),local_pull_visual_QA_and_native_TeX_compile_pending=True,time=time.time()))
 wr('PIPELINE_STATUS.json',dict(stage='REMOTE_RESULTS_READY_LOCAL_VERIFICATION_PENDING',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:wr('PIPELINE_FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
