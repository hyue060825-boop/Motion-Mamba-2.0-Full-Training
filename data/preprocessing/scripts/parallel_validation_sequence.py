"""Finish in-flight checks, finalize orchestration, then run 16 before 32.

Only new executor code is replaced, only after all its current processes exit.
Frozen serial/solver files and all old fingerprints are left unchanged.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil
from g1_pipeline.common import ROOT,sha,write_json,atomic_bytes,now
from parallel_compare import compare

NS=['benchmark_equivalence_w1_v1','benchmark_equivalence_w4_v1']
EXPECTED='3442f443f80f1b067d3fd6cd1e6a6450aca6af8ac743745860815bcebd6c0b50'
PENDING=Path('/tmp/g1_parallel_runner_pending.py')

def stage(name,**extra):
 write_json(ROOT/'audit/parallel_validation_sequence_status.json',{'time':now(),'pid':os.getpid(),'phase':name,**extra})
 print(name,flush=True)

def live_executors():
 result=[]
 for p in psutil.process_iter(['pid','cmdline']):
  cmd=p.info.get('cmdline') or []
  if 'g1_pipeline.parallel_runner' in cmd:result.append({'pid':p.pid,'command':cmd})
 return result

def command(script,env=None):
 log=(ROOT/'logs'/('sequence_'+Path(script).stem+'.log')).open('ab')
 p=subprocess.Popen([sys.executable,script],cwd=ROOT,env=env or os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
 stage('RUNNING_'+Path(script).stem,child_pid=p.pid)
 code=p.wait();log.close()
 if code:raise RuntimeError('VALIDATION_STAGE_FAILED:'+script+':'+str(code))

def run():
 stage('WAIT_INITIAL_FULL_CLIP_EQUIVALENCE')
 expected_pending=sha(PENDING)
 while True:
  finished=all((ROOT/'work'/ns/'summary.json').exists() for ns in NS)
  if finished and not live_executors():break
  time.sleep(10)
 for ns in NS:
  assert json.loads((ROOT/'work'/ns/'summary.json').read_text())['status']=='COMPLETE'
  if compare(ns)['status']!='PASS':raise ValueError('INITIAL_NUMERICAL_EQUIVALENCE_FAILED:'+ns)
 stage('INITIAL_EQUIVALENCE_PASS')
 # Accounting/stop-request lifecycle/serial import only. The single-clip helper
 # and frozen source computation remain exactly the verified version.
 target=ROOT/'g1_pipeline/parallel_runner.py'
 assert sha(target)==EXPECTED and sha(PENDING)==expected_pending and not live_executors()
 history=ROOT/'audit/parallel_executor_history'
 for name in ('parallel_recovery_test','parallel_guard_tests'):
  p=ROOT/'audit'/(name+'.json');h=sha(p)
  atomic_bytes(history/(name+'_'+h+'.json'),p.read_bytes())
 compile(PENDING.read_text(),str(target),'exec')
 atomic_bytes(target,PENDING.read_bytes())
 h=sha(target);atomic_bytes(history/('parallel_runner_'+h+'.py'),target.read_bytes())
 write_json(history/'final_executor_upgrade.json',{'time':now(),'from':EXPECTED,'to':h,
  'path':str((history/('parallel_runner_'+h+'.py')).relative_to(ROOT)),
  'reason':'Correct resume throughput accounting, consume explicit stop on resume, verify and reference completed serial pilot artifacts',
  'active_executor_processes_at_change':[],'frozen_computation_changed':False,
  'old_fingerprints_rewritten':False})
 env=os.environ.copy();env['PARALLEL_VALIDATION_VERSION']='v2'
 command('scripts/parallel_recovery_test.py',env)
 command('scripts/parallel_guard_tests.py',env)
 command('scripts/parallel_boundary_reuse_test.py',env)
 stage('FINAL_EXECUTOR_SAFETY_TESTS_PASS')
 command('scripts/parallel_benchmark.py',env)
 stage('BENCHMARKS_COMPLETE_PILOT_REVIEW_STILL_REQUIRED')

if __name__=='__main__':
 try:run()
 except BaseException as exc:
  stage('FAILED',error=repr(exc));raise
