"""Apply bounded retry and pilot schema compatibility after all measured executors exit."""
import fcntl
import json
import os
import subprocess
import sys
import time
import psutil
from g1_pipeline.common import ROOT,sha,atomic_bytes,write_json,now
from pathlib import Path


def live_executors():
 return [p.info['pid'] for p in psutil.process_iter(['pid','cmdline'])
         if 'g1_pipeline.parallel_runner' in (p.info.get('cmdline') or [])]

def run():
 status=ROOT/'audit/parallel_retry_upgrade_status.json'
 def state(phase,**extra):write_json(status,{'time':now(),'pid':os.getpid(),'phase':phase,**extra})
 with (ROOT/'audit/parallel_executor_update.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('EXECUTOR_UPGRADE_ALREADY_RUNNING')
  plan=json.loads((ROOT/'audit/parallel_retry_upgrade_plan.json').read_text())
  state('WAIT_MEASURED_16_32_AND_IDLE_EXECUTOR')
  paths=[ROOT/f'audit/benchmark_throughput_w{w}_v1_resources.json' for w in [16,32]]
  while True:
   complete=all(p.exists() and json.loads(p.read_text())['status']=='PASS' for p in paths)
   if complete and (ROOT/'audit/parallel_worker_selection.json').exists() and not live_executors():break
   time.sleep(10)
  target=ROOT/'g1_pipeline/parallel_runner.py';pending=Path(plan['pending_path'])
  assert sha(target)==plan['current_executor_sha256'] and sha(pending)==plan['pending_executor_sha256']
  assert not live_executors()
  history=ROOT/'audit/parallel_executor_history'
  oldhash=sha(target);oldpath=history/('parallel_runner_'+oldhash+'.py')
  if not oldpath.exists():atomic_bytes(oldpath,target.read_bytes())
  assert sha(oldpath)==oldhash
  compile(pending.read_text(),str(target),'exec')
  atomic_bytes(target,pending.read_bytes());newhash=sha(target)
  newpath=history/('parallel_runner_'+newhash+'.py');atomic_bytes(newpath,target.read_bytes())
  report={'time':now(),'status':'APPLIED_PENDING_FOCUSED_TEST','from_sha256':oldhash,'to_sha256':newhash,
          'from_source_path':str(oldpath.relative_to(ROOT)),'to_source_path':str(newpath.relative_to(ROOT)),
          'active_executors_at_update':[], 'change':'Bounded transient I/O retry (maximum two attempts) and original pilot evaluator schema compatibility; schema/IK/QA stay terminal; solver unchanged.',
          'retarget_helper_sha256':sha(ROOT/'g1_pipeline/parallel_clip.py'),
          'measured_benchmark_executor_sha256':oldhash,'measured_benchmark_report_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},
          'new_version_fresh_throughput_remeasured':False,'old_run_fingerprints_rewritten':False,'frozen_algorithm_changed':False}
  path=ROOT/'audit/parallel_retry_executor_upgrade.json';write_json(path,report)
  log=(ROOT/'logs/parallel_io_retry_test.log').open('ab')
  p=subprocess.Popen([sys.executable,'scripts/parallel_io_retry_test.py'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
  state('RUNNING_FOCUSED_IO_RETRY_TEST',child_pid=p.pid)
  code=p.wait();log.close()
  if code:state('FAILED_IO_RETRY_TEST',returncode=code);raise RuntimeError('IO_RETRY_TEST_FAILED')
  result=json.loads((ROOT/'audit/parallel_io_retry_test.json').read_text())
  assert result['status']=='PASS' and result['executor_sha256']==sha(target)
  state('RUNNING_FOCUSED_PILOT_SCHEMA_TEST')
  with (ROOT/'logs/parallel_pilot_schema_controller.log').open('ab') as schema_log:
   schema_run=subprocess.run([sys.executable,'scripts/parallel_pilot_schema_test.py'],cwd=ROOT,env=os.environ.copy(),stdout=schema_log,stderr=subprocess.STDOUT)
  if schema_run.returncode:state('FAILED_PILOT_SCHEMA_TEST',returncode=schema_run.returncode);raise RuntimeError('PILOT_SCHEMA_TEST_FAILED')
  schema=json.loads((ROOT/'audit/parallel_pilot_schema_test.json').read_text())
  assert schema['status']=='PASS' and schema['executor_sha256']==sha(target)
  report['schema_test_sha256']=sha(ROOT/'audit/parallel_pilot_schema_test.json')
  report['status']='VERIFIED';report['focused_test_sha256']=sha(ROOT/'audit/parallel_io_retry_test.json');report['verified_at']=now()
  write_json(path,report);state('COMPLETE',executor_sha256=sha(target))
  print('Bounded I/O retry upgrade and focused verification complete',flush=True)

if __name__=='__main__':run()
