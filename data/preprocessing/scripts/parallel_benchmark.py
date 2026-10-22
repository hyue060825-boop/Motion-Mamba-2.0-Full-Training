"""Sequential 16/32 runs with host/process resource sampling and exact checks."""
import json
import os
import subprocess
import sys
import time
import sqlite3
from collections import defaultdict
import psutil
from g1_pipeline.common import ROOT,sha,write_json,read_jsonl,now
from parallel_compare import compare


def one(workers):
 ns=f'benchmark_throughput_w{workers}_v1';base=ROOT/'work'/ns
 panel=ROOT/'audit/parallel_throughput_panel.json'
 if base.exists():
  reportpath=ROOT/f'audit/{ns}_resources.json'
  if not reportpath.exists() or json.loads((base/'summary.json').read_text())['status']!='COMPLETE':
   raise ValueError('EXISTING_INCOMPLETE_BENCHMARK_REQUIRES_PROCESS_STATUS_AUDIT')
  previous=json.loads(reportpath.read_text())
  if previous['panel_sha256']!=sha(panel) or previous['bindings']['run']!=sha(base/'run.json') or previous['bindings']['ledger']!=sha(base/'results.jsonl'):
   raise ValueError('COMPLETED_BENCHMARK_EVIDENCE_CHANGED')
  comparison=compare(ns)
  if comparison['status']!='PASS':raise ValueError('COMPLETED_BENCHMARK_COMPARISON_NOT_PASSED')
  newhash=sha(ROOT/f'audit/{ns}_comparison.json')
  if previous['status']!='PASS' or previous['bindings']['comparison']!=newhash:
   historical=ROOT/'audit/parallel_executor_history'/(reportpath.stem+'_'+sha(reportpath)+'.json')
   historical.write_bytes(reportpath.read_bytes())
   previous['status']='PASS';previous['bindings']['comparison']=newhash
   previous['comparison_revalidated_without_rerunning_solver']=True
   write_json(reportpath,previous)
  print({'namespace':ns,'reuse_complete_measurement':True,'frames_per_sec':previous['frames_per_sec']},flush=True)
  return previous
 log=(ROOT/'logs'/(ns+'.log')).open('wb')
 command=[sys.executable,'-m','g1_pipeline.parallel_runner','run','--namespace',ns,'--panel',str(panel.relative_to(ROOT)),'--workers',str(workers)]
 start=time.monotonic();wall_start=now();io0=psutil.disk_io_counters();cpu0=psutil.cpu_times();mem0=psutil.virtual_memory()
 p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 samples=[];per_pid={};chrome={}
 for process in psutil.process_iter(['pid','name']):
  if 'chrome' in (process.info['name'] or '').lower():
   try:chrome[process.pid]={'initial_cpu':sum(process.cpu_times()[:2]),'final_cpu':sum(process.cpu_times()[:2])}
   except psutil.Error:pass
 while p.poll() is None:
  total_rss=0
  try:processes=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
  except psutil.Error:processes=[]
  for process in processes:
   try:
    mem=process.memory_info();cpu=process.cpu_times();io=process.io_counters();total_rss+=mem.rss
    old=per_pid.setdefault(str(process.pid),{'max_rss_bytes':0})
    old.update(cpu_seconds=cpu.user+cpu.system,read_bytes=io.read_bytes,write_bytes=io.write_bytes,
               read_chars=io.read_chars,write_chars=io.write_chars,
               max_rss_bytes=max(old['max_rss_bytes'],mem.rss),threads=process.num_threads())
   except psutil.Error:pass
  for pid,v in chrome.items():
   try:v['final_cpu']=sum(psutil.Process(pid).cpu_times()[:2])
   except psutil.Error:pass
  samples.append({'sec':time.monotonic()-start,'total_rss_bytes':total_rss,'load':list(os.getloadavg()),
                  'available_memory_bytes':psutil.virtual_memory().available,'processes':len(processes)})
  time.sleep(2)
 wall=time.monotonic()-start;io1=psutil.disk_io_counters();log.close()
 if p.returncode:raise RuntimeError('BENCHMARK_COORDINATOR_FAILED:'+ns)
 summary=json.loads((base/'summary.json').read_text());assert summary['status']=='COMPLETE'
 with sqlite3.connect(base/'state.sqlite') as c:
  outputs=[json.loads(row[0]) for row in c.execute('SELECT output_json FROM items')]
 executions=[json.loads(open(o['execution_path']).read()) for o in outputs]
 phases=defaultdict(float)
 for e in executions:
  for key,val in e.get('phase_seconds',{}).items():phases[key]+=val
 frames=summary['frames'];comparison=compare(ns)
 report={'namespace':ns,'workers':workers,'status':'PASS' if comparison['status']=='PASS' else 'FAIL',
         'started_at':wall_start,'finished_at':now(),'actual_process_wall_sec_including_preflight':wall,
         'frames':frames,'frames_per_sec':frames/wall,'source_motion_hours_per_hour':frames/25/wall,
         'counts':summary['counts'],'complete_clips':summary['completed'],'panel_sha256':sha(panel),
         'phase_seconds_sum_across_workers':dict(phases),
         'phase_timing_note':'wrapper timers around unchanged functions; sums overlap only parse frames_at_25hz/native read calls within different original entry functions; process wall includes initialization and validation',
         'cpu_seconds_sum_worker_rusage':sum(e['cpu_sec'] for e in executions),
         'peak_sampled_process_tree_rss_bytes':max(s['total_rss_bytes'] for s in samples),
         'sum_per_clip_peak_rss_is_not_simultaneous':sum(e['peak_rss_bytes'] for e in executions),
         'worker_io_blocks_512_bytes':{'read':sum(e['read_blocks'] for e in executions),'write':sum(e['write_blocks'] for e in executions)},
         'host_io_bytes_including_other_tasks':{'read':io1.read_bytes-io0.read_bytes,'write':io1.write_bytes-io0.write_bytes},
         'chrome_mean_cpu_cores':sum(v['final_cpu']-v['initial_cpu'] for v in chrome.values())/wall,
         'cache_policy':'existing source/mapping/model files and OS page cache; fresh full solver/state per clip; no trajectory cache; 16 precedes 32 without dropping shared host caches',
         'render_included':False,'gpu_solver':False,'samples':samples,'sampled_processes':per_pid,
         'clip_wall_seconds':[e['wall_sec'] for e in executions],
         'bindings':{'run':sha(base/'run.json'),'ledger':sha(base/'results.jsonl'),'comparison':sha(ROOT/f'audit/{ns}_comparison.json')}}
 write_json(ROOT/f'audit/{ns}_resources.json',report)
 print({k:report[k] for k in ('namespace','status','frames_per_sec','actual_process_wall_sec_including_preflight','chrome_mean_cpu_cores')},flush=True)
 if comparison['status']!='PASS':raise ValueError('BENCHMARK_OUTPUT_DIFFERENCE')
 return report

if __name__=='__main__':
 for name in ('benchmark_equivalence_w1_v1','benchmark_equivalence_w4_v1'):
  assert json.loads((ROOT/f'audit/{name}_comparison.json').read_text())['status']=='PASS'
 assert json.loads((ROOT/'audit/parallel_recovery_test.json').read_text())['status']=='PASS'
 assert json.loads((ROOT/'audit/parallel_guard_tests.json').read_text())['status']=='PASS'
 a=one(16);b=one(32)
 speedup=b['frames_per_sec']/a['frames_per_sec']
 selected=32 if speedup>=1.15 else 16
 write_json(ROOT/'audit/parallel_worker_selection.json',{'status':'MEASURED','selected_workers':selected,
  'speedup_32_over_16':speedup,'selection_reason':'Use 32 only if measured gain is at least 15%; otherwise 16 reduces contention',
  'tested_workers':[16,32],'test_48':False,
  '48_reason':'Optional; at most 64 physical cores and substantial observed foreign Chrome load. Keep measured 16/32 configuration.',
  'bound_resource_reports':{str(ROOT/f'audit/{r["namespace"]}_resources.json'):sha(ROOT/f'audit/{r["namespace"]}_resources.json') for r in [a,b]}})
