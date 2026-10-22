from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,json,hashlib,subprocess,shutil,time,fcntl,ast
r=Path(artifact_path('g1_native112k_future_interaction_20261011_d06c37e3c2'))
py=artifact_path('python_9f8003379b')
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
for name in ['future_goal.py','train_future.py']:ast.parse((r/name).read_text())
assert read(r/'acceleration_v4/FULL_UPDATE_GATE.json')['passed']
assert read(r/'acceleration_v4/FULL_TIMING.json')['complete']
assert read(r/'LATEST.json')['stage_step']==1000
assert hashlib.sha256((r/'config.json').read_bytes()).hexdigest()=='b20fa0fed92dac5ca8ac4ddbbe0169a3da4eeb386a4eb690a00e0193839471c4'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not apps,('Other GPU workloads present; do not start',apps)
# Failed distributed probe killed rank0 before atexit. Close its stale budget
# conservatively at the next probe start, an upper bound on its actual lifetime.
with (r/'budget/budget.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 active=read(r/'budget/ACTIVE.json')
 if '2979424' in active:
  try:os.kill(2979424,0)
  except ProcessLookupError:pass
  else:raise RuntimeError('Old profiling process is still alive')
  entry=active.pop('2979424')
  end=(r/'acceleration_v4/ACTIVE.json').stat().st_mtime
  assert end>entry['start']
  receipt=dict(name=entry['name']+':failed_conservative_recovery',seconds=end-entry['start'],devices=entry['devices'],device=entry['device'],budget_recovery=True,end_is_upper_bound=True,pid=2979424)
  with (r/'budget/GPU_TIME.jsonl').open('a') as g:g.write(json.dumps(receipt)+'\n')
  write(r/'budget/ACTIVE.json',active)
  write(r/'acceleration_v4/BUDGET_RECOVERY.json',receipt)
if (r/'FAILURE.json').exists():
 shutil.copy2(r/'FAILURE.json',r/'acceleration_v4/FAILURE.before_resumption.json')
 (r/'FAILURE.json').unlink()
log=r/'logs/controller_accelerated_resume.log'
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with log.open('a') as f:
 child=subprocess.Popen([py,'controller_future.py'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=child.pid,time=time.time(),resume_step=1000,checkpoint=str(r/'train/checkpoint1000.pt'),pool=str(r/'pools/g1/POOL.json'),config_unchanged=True,files={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in ['future_goal.py','train_future.py']},timing=read(r/'acceleration_v4/FULL_TIMING.json')['medians'],log=str(log))
write(r/'acceleration_v4/RESUMPTION.json',receipt)
print(json.dumps(receipt,indent=2))
