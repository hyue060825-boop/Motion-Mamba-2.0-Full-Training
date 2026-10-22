from pathlib import Path
import os,sys,json,time,subprocess,fcntl,traceback
from execute import R,wr
from controller import free,acquire,PY
O=R/'diagnosis/native122_process_20261021';O.mkdir(parents=True,exist_ok=True)
lock=(O/'resource.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for card in free():
 files=acquire(card)
 if not files:continue
 try:
  wr(O/'LAUNCH.json',dict(pid=os.getpid(),card=card,script=str(R/'audit_process_122k.py'),time=time.time()))
  with (O/'audit.log').open('a') as log:
   p=subprocess.Popen([PY,str(R/'audit_process_122k.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   wr(O/'ACTIVE_CHILD.json',dict(pid=p.pid,card=card,time=time.time()));p.wait()
  if p.returncode:raise RuntimeError(f'Process diagnostic exited with{p.returncode}; see failure evidence')
  print('Bounded GPU audit completed')
 finally:
  for f in files:f.close()
 break
else:raise RuntimeError('No actually free and unlocked GPU; no other job was interrupted')
