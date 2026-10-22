from pathlib import Path
import os,sys,time,subprocess,fcntl
from execute import R,wr
from controller import free,acquire,PY
O=R/'diagnosis/native122_process_20261021'
lock=(O/'clarity_resource.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for card in free():
 files=acquire(card)
 if not files:continue
 try:
  with (O/'command_clarity.log').open('a') as log:
   p=subprocess.Popen([PY,str(R/'audit_command_clarity.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   wr(O/'CLARITY_PID.json',dict(pid=p.pid,card=card,time=time.time()));rc=p.wait()
  assert rc==0,rc
  print('Command clarity diagnostic complete')
 finally:
  for f in files:f.close()
 break
else:raise RuntimeError('No free unlocked GPU; no job interrupted')
