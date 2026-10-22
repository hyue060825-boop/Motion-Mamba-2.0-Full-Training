"""Low-priority exact external acceleration admission, only after generation releases GPUs."""
from pathlib import Path
import os,sys,time,subprocess,fcntl,traceback
from execute import R,rd,wr
from controller import free,acquire,PY
from retime import METHODS
def main():
 out=R/'retiming';out.mkdir(exist_ok=True);lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wr(out/'CONTROLLER_PID.json',dict(pid=os.getpid(),script=str(Path(__file__).resolve()),time=time.time()))
 jobs=[(m,t) for t in ['nominal40','nominal20','nominal9','short52'] for m in METHODS];active={};failed=[]
 # Generation owns queue priority; do not occupy cards needed by unlaunched formal jobs.
 while not (R/'GENERATION_COMPLETE.json').exists():
  wr(out/'STATUS.json',dict(stage='WAITING_ALL_FORMAL_GENERATION_RELEASE',time=time.time()));time.sleep(20)
 queue=[k for k in jobs if not (out/k[0]/k[1]/'COMPLETE.json').exists()]
 while queue or active:
  for k,v in list(active.items()):
   rc=v['p'].poll()
   if rc is None:continue
   v['log'].close()
   for f in v['locks']:f.close()
   del active[k]
   if rc:failed.append(list(k));print('RETIMING_FAILED',k,rc,flush=True)
  for card in free():
   if not queue:break
   files=acquire(card)
   if not files:continue
   m,t=queue.pop(0);d=out/m/t;d.mkdir(parents=True,exist_ok=True);log=(d/'worker.log').open('a')
   p=subprocess.Popen([PY,str(R/'retime.py'),'--model',m,'--task',t],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   active[(m,t)]=dict(p=p,log=log,locks=files,card=card)
  wr(out/'STATUS.json',dict(stage='EXACT_CORE_GATING_AND_RETIMING',active=[dict(model=k[0],task=k[1],pid=v['p'].pid,card=v['card']) for k,v in active.items()],queued=[list(k) for k in queue],failed=failed,time=time.time()))
  if queue or active:time.sleep(10)
 if failed:wr(out/'FAILURE.json',dict(jobs=failed,time=time.time()))
 else:
  if not (R/'TEXT_UPDATE_LATENCY.json').exists():
   while True:
    got=None
    for card in free():
     files=acquire(card)
     if files:got=(card,files);break
    if got:
     card,files=got
     try:
      with (out/'text_update.log').open('a') as log:subprocess.run([PY,str(R/'text_update_latency.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True)
      break
     finally:
      for f in files:f.close()
    time.sleep(15)
  wr(out/'COMPLETE.json',dict(passed=True,jobs=len(jobs),text_update_timing_verified=True,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:wr(R/'retiming'/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
