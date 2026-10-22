"""Wait for this task's live input stager, then run the authorized complete queue."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,subprocess,sys,time
import psutil
from g1_pipeline.common import ROOT,write_json,now
from g1_pipeline.kinematic_policy import BASE,load

def run(stager_pid):
 if not (BASE/'input_release.json').exists():
  process=psutil.Process(stager_pid)
  if process.cmdline()[-1:]!=['scripts/kinematic_prepare.py']:raise ValueError('STAGER_IDENTITY_CHANGED')
  birth=process.create_time()
  while not (BASE/'input_release.json').exists():
   try:live=process.is_running() and process.status()!=psutil.STATUS_ZOMBIE and process.create_time()==birth
   except psutil.NoSuchProcess:live=False
   if not live:raise RuntimeError('INPUT_STAGER_EXITED_BEFORE_SEAL')
   write_json(BASE/'continuation_status.json',{'time':now(),'phase':'WAITING_FOR_LIVE_INPUT_STAGER','pid':os.getpid(),'stager_pid':stager_pid,'stager_create_time':birth})
   time.sleep(30)
 for command in [['scripts/kinematic_release.py','run'],['scripts/kinematic_postprocess.py']]:
  write_json(BASE/'continuation_status.json',{'time':now(),'phase':'EXECUTING','pid':os.getpid(),'command':command})
  result=subprocess.run([sys.executable,*command],cwd=ROOT,env=os.environ.copy())
  if result.returncode:raise RuntimeError('AUTHORIZED_EXECUTION_FAILED:'+repr(command))
 write_json(BASE/'continuation_status.json',{'time':now(),'phase':'FINAL_SEMANTIC_REVIEW_REQUIRED','ready':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stager-pid',required=True,type=int);a=p.parse_args();run(a.stager_pid)
