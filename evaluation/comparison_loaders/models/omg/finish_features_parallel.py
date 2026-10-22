"""Run unchanged CPU scoring in eight independent processes after GPU work.

Pause only this evaluation's scheduler, finish its active CPU children, and
resume in finally. This prevents duplicate writers without modifying any
frozen sampler/evaluator source, tensor batch, precision or thread count.
"""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,signal,subprocess,time,json,concurrent.futures,fcntl
R=Path(artifact_path('g1_external_matched100k_evaluation_20261019_aa0c66c5b2'))
PY=artifact_path('python_9f8003379b')
lock=(R/'feature_finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s=json.loads((R/'STATUS.json').read_text());assert s['complete_jobs']==s['total_jobs']==224 and not s['active']
pid=s['controller_pid'];assert Path(f'/proc/{pid}/cwd').resolve()==R
assert (R/'controller.py').as_posix() in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
assert os.cpu_count()>=32 and os.getloadavg()[0]<32
receipt=dict(controller_pid=pid,unchanged_cpu_score_source=True,processes=8,threads_per_process=2,time=time.time(),gpu_jobs_finished=224)
os.kill(pid,signal.SIGSTOP)
try:
 deadline=time.monotonic()+180
 while True:
  children=subprocess.check_output(['ps','--ppid',str(pid),'-o','pid=,stat=,args='],text=True).splitlines()
  live=[line for line in children if 'score_job.py' in line and not line.split()[1].startswith('Z')]
  if not live:break
  assert time.monotonic()<deadline,'Original scoring children did not finish'
  time.sleep(1)
 pending=[p.parent for p in sorted((R/'evaluation').glob('*/*/shard*/COMPLETE.json')) if not (p.parent/'WINDOW_SCORE_COMPLETE.json').exists()]
 receipt['remaining_jobs']=len(pending)
 def score(folder):
  with (folder/'score_parallel.log').open('a') as log:
   subprocess.run([PY,'-u',str(R/'score_job.py'),'--job',str(folder)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
  assert (folder/'WINDOW_SCORE_COMPLETE.json').exists()
  return str(folder)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for folder in pool.map(score,pending):print('SCORED',folder,flush=True)
 assert len(list((R/'evaluation').glob('*/*/shard*/WINDOW_SCORE_COMPLETE.json')))==224
 receipt.update(status='COMPLETE_UNCHANGED_FROZEN_CPU_SCORING',completed_time=time.time())
finally:
 os.kill(pid,signal.SIGCONT)
 (R/'FEATURE_PARALLEL_FINISH.json').write_text(json.dumps(receipt,indent=2))
print('RESUMED_CONTROLLER',pid,flush=True)
