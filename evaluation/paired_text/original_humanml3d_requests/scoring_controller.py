"""CPU-only scoring queue, consumes complete saved generation jobs exactly once."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,time,subprocess,fcntl,traceback
from execute import R,MODELS,rd,wr,sha
PY=artifact_path('python_9f8003379b')
def main():
 out=R/'scoring';out.mkdir(exist_ok=True);lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wr(out/'CONTROLLER_PID.json',dict(pid=os.getpid(),script=str(Path(__file__).resolve()),cwd=str(Path.cwd()),CPU_only=True,time=time.time()))
 jobs=[(m,t) for t in ['nominal40','nominal20','nominal9','short52'] for m in MODELS];active={};failed=set()
 while True:
  for key,v in list(active.items()):
   rc=v['p'].poll()
   if rc is None:continue
   v['log'].close();del active[key]
   if rc:failed.add(key);print('SCORE_FAILED',key,rc,flush=True)
   else:print('SCORE_COMPLETE',key,flush=True)
  completed=[k for k in jobs if (out/'models'/k[0]/k[1]/'COMPLETE.json').exists()]
  if (out/'REFERENCE_COMPLETE.json').exists():
   ready=[k for k in jobs if k not in completed and k not in active and k not in failed and (R/'models'/k[0]/k[1]/'GENERATION_COMPLETE.json').exists()]
   for m,t in ready[:max(0,3-len(active))]:
    d=out/'models'/m/t;d.mkdir(parents=True,exist_ok=True);log=(d/'worker.log').open('a')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    p=subprocess.Popen([PY,str(R/'score.py'),'--model',m,'--task',t],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[(m,t)]=dict(p=p,log=log);print('CPU_SCORE_STARTED',m,t,p.pid,flush=True)
  wr(out/'STATUS.json',dict(stage='SCORING_COMPLETED_SAVED_GEOMETRY',completed=[list(k) for k in completed],active=[dict(model=k[0],task=k[1],pid=v['p'].pid) for k,v in active.items()],failed=[list(k) for k in failed],total=36,time=time.time()))
  if len(completed)==len(jobs):
   with (out/'finalize.log').open('a') as log:subprocess.run([PY,str(R/'finalize.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
   break
  if failed and not active:
   wr(out/'FAILURE.json',dict(stage='SCORING_FAILURE_REQUIRES_EXECUTION_REPAIR',jobs=[list(k) for k in failed],time=time.time()))
  time.sleep(20)
if __name__=='__main__':
 try:main()
 except BaseException:wr(R/'scoring'/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
