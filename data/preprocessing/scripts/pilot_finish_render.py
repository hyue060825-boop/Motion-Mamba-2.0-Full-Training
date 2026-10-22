"""Wait for the original runner lock, then fill missing full-clip review videos.

No solver reruns. One renderer. Main serial runner retains its lock through its
own renderer and evaluator, so there is no shared video-manifest race.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['MUJOCO_GL']='egl'
import fcntl
import json
import time
from pathlib import Path
from g1_pipeline.common import ROOT,read_jsonl,sha,write_json,now


def run():
 ns='pilot_hiphi100_native_v1_eb91dfef770f';base=ROOT/'work'/ns
 with (base/'review_render_coordinator.lock').open('a+') as own:
  try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('REVIEW_RENDER_COORDINATOR_ALREADY_RUNNING')
  with (base/'run.lock').open('a+') as lock:
   while True:
    try:
     fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:
     write_json(ROOT/'audit/pilot_finish_render_status.json',{'time':now(),'pid':os.getpid(),'phase':'WAIT_ORIGINAL_RUNNER_LOCK'})
     time.sleep(10)
   records=read_jsonl(base/'results.jsonl')
   if len(records)!=len({r['sample_id'] for r in records}) or len(records)!=100:raise ValueError('ORIGINAL_PILOT_NOT_COMPLETE')
   usable=[r for r in records if r.get('output_paths')]
   vp=ROOT/'videos'/ns/'manifest.json'
   before=json.loads(vp.read_text())['videos'] if vp.exists() else []
   existing={r['sample_id'] for r in before}
   pending=[r for r in usable if r['sample_id'] not in existing]
   write_json(ROOT/'audit/pilot_finish_render_status.json',{'time':now(),'pid':os.getpid(),'phase':'RENDER_MISSING','existing':len(existing),'missing':len(pending)})
   from g1_pipeline.render_frozen_review import run as render
   start=time.monotonic()
   render(ns,[r['sample_id'] for r in usable],source='hiphi')
   elapsed=time.monotonic()-start
   after=json.loads(vp.read_text())['videos']
   assert len(after)==len(usable)
   for v in after:
    assert sha(ROOT/v['path'])==v['sha256'] and sha(ROOT/v['sheet_path'])==v['sheet_sha256']
   # State records only rendering; sampled visual review is still a separate step.
   result={'time':now(),'pid':os.getpid(),'phase':'RENDER_COMPLETE_REVIEW_PENDING','namespace':ns,
           'render_workers':1,'rendered_total':len(after),'reused_videos':len(existing),'new_videos':len(pending),
           'new_frames':sum(r['frames'] for r in pending),'elapsed_sec':elapsed,
           'frames_per_sec_for_new_outputs':sum(r['frames'] for r in pending)/elapsed,
           'manifest_sha256':sha(vp),'solver_reruns':0,'visual_review_completed':False}
   write_json(ROOT/'audit/pilot_finish_render_status.json',result)
   print(result,flush=True)
if __name__=='__main__':run()
