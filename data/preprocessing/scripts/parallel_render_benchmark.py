"""Measure unchanged full-clip renderer separately, at concurrency one.

Uses the exact four-clip equivalence outputs in separate benchmark namespaces.
Waits for original pilot's run lock, which includes its final rendering phase.
No solver execution, altered frame count, or production manifest is involved.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['MUJOCO_GL']='egl'
import fcntl
import json
import resource
import subprocess
import threading
import time
from pathlib import Path
import psutil
from g1_pipeline.common import ROOT,read_jsonl,write_json,write_jsonl,sha,now


def run():
 status=ROOT/'audit/parallel_render_benchmark_status.json'
 def state(phase,**extra):write_json(status,{'time':now(),'pid':os.getpid(),'phase':phase,**extra})
 with (ROOT/'audit/parallel_render_benchmark.lock').open('a+') as own:
  try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('RENDER_BENCHMARK_ALREADY_RUNNING')
  eq='benchmark_equivalence_w4_v1';check=ROOT/f'audit/{eq}_comparison.json'
  state('WAIT_EXACT_EQUIVALENCE')
  while not check.exists():time.sleep(10)
  assert json.loads(check.read_text())['status']=='PASS'
  original=ROOT/'work/pilot_hiphi100_native_v1_eb91dfef770f'
  with (original/'run.lock').open('a+') as serial:
   state('WAIT_ORIGINAL_PILOT_RENDER_LOCK')
   while True:
    try:fcntl.flock(serial,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:time.sleep(10)
   rows=read_jsonl(ROOT/'work'/eq/'results.jsonl')
   for r in rows:
    for p,h in r['output_hashes'].items():assert sha(ROOT/p)==h
   from g1_pipeline.render_frozen_review import run as render
   samples=[];stop=threading.Event()
   def monitor():
    root=psutil.Process(os.getpid())
    while not stop.is_set():
     rss=0;cpu=0.;io_read=io_write=0
     for p in [root]+root.children(recursive=True):
      try:
       rss+=p.memory_info().rss;c=p.cpu_times();cpu+=c.user+c.system
       io=p.io_counters();io_read+=io.read_bytes;io_write+=io.write_bytes
      except psutil.Error:pass
     try:
      gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5).stdout.strip()
     except (OSError,subprocess.TimeoutExpired):gpu=None
     samples.append({'time':now(),'rss_bytes':rss,'observed_cpu_seconds':cpu,'io_read_bytes':io_read,'io_write_bytes':io_write,'gpu':gpu})
     stop.wait(2)
   state('RENDERING_FULL_CLIPS',clips=len(rows),frames=sum(r['frames'] for r in rows))
   thread=threading.Thread(target=monitor,daemon=True);thread.start()
   start=time.monotonic();groups=[];before_child=resource.getrusage(resource.RUSAGE_CHILDREN);before_self=resource.getrusage(resource.RUSAGE_SELF)
   try:
    for source,short in [('humanml3d','hml'),('hiphi','hiphi')]:
     group=[r for r in rows if r['dataset']==source]
     ns='benchmark_render_'+short+'_v1';base=ROOT/'work'/ns
     if base.exists():raise ValueError('RENDER_BENCHMARK_NAMESPACE_EXISTS')
     base.mkdir();write_jsonl(base/'results.jsonl',group)
     write_json(base/'run.json',{'scope':'isolated_render_benchmark','source_namespace':eq,'solver_reruns':0,'training_release':False})
     t=time.monotonic();videos=render(ns,[r['sample_id'] for r in group],source=source);elapsed=time.monotonic()-t
     frames=sum(v['frames'] for v in videos)
     assert len(videos)==len(group) and frames==sum(r['frames'] for r in group)
     vp=ROOT/'videos'/ns/'manifest.json'
     groups.append({'source':source,'namespace':ns,'clips':len(group),'frames':frames,'wall_sec':elapsed,'frames_per_sec':frames/elapsed,'manifest_sha256':sha(vp)})
   finally:stop.set();thread.join(timeout=10)
   wall=time.monotonic()-start;child=resource.getrusage(resource.RUSAGE_CHILDREN);ownusage=resource.getrusage(resource.RUSAGE_SELF)
   frames=sum(g['frames'] for g in groups)
   solve=json.loads((ROOT/'work'/eq/'summary.json').read_text())
   result={'status':'MEASURED','time':now(),'renderer_workers':1,'encoder_threads':2,'full_clips':len(rows),'frames':frames,
           'wall_sec':wall,'render_frames_per_sec':frames/wall,'source_motion_hours_per_hour':frames/25/wall,
           'groups':groups,'peak_sampled_process_tree_rss_bytes':max(s['rss_bytes'] for s in samples),
           'cpu_seconds_self':ownusage.ru_utime+ownusage.ru_stime-before_self.ru_utime-before_self.ru_stime,
           'cpu_seconds_children':child.ru_utime+child.ru_stime-before_child.ru_utime-before_child.ru_stime,
           'io_blocks_512_bytes':{'read':ownusage.ru_inblock+child.ru_inblock-before_self.ru_inblock-before_child.ru_inblock,
                                 'write':ownusage.ru_oublock+child.ru_oublock-before_self.ru_oublock-before_child.ru_oublock},
           'four_clip_compute_qa_plus_render_seconds_sum':solve['wall_sec']+wall,
           'four_clip_compute_qa_plus_render_frames_per_sec':frames/(solve['wall_sec']+wall),
           'timing_scope':'Sum of measured four-worker solve/export/QA run and one-renderer full-video run for the same four complete clips. Idle waiting for pilot lock excluded; not a measured inclusive 64-clip run.',
           'gpu_device_policy':'Existing EGL default selection; no multi-GPU distribution claimed. nvidia-smi records include other host jobs.',
           'cache_policy':'Fresh video namespaces, no video reuse; original full trajectories reused without solver reruns',
           'qa_representation_pass_comparison':sha(check),'renderer_sha256':sha(ROOT/'g1_pipeline/render_frozen_review.py'),
           'samples':samples,'production_release':False}
   write_json(ROOT/'audit/parallel_render_benchmark.json',result);state('COMPLETE',report='audit/parallel_render_benchmark.json')
   print({k:result[k] for k in ['status','frames','wall_sec','render_frames_per_sec','peak_sampled_process_tree_rss_bytes']})

if __name__=='__main__':run()
