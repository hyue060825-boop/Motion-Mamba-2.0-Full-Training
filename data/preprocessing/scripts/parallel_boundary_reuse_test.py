"""Verify real stop/resume CLI and import the existing HML100 without solving."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha
from parallel_compare import compare


def wait_for(fn,limit=180):
 end=time.monotonic()+limit
 while time.monotonic()<end:
  value=fn()
  if value:return value
  time.sleep(.1)
 raise TimeoutError('BOUNDARY_TEST_OBSERVATION_TIMEOUT')

def run():
 ns='benchmark_boundary_v2';base=ROOT/'work'/ns
 if base.exists():raise ValueError('BOUNDARY_TEST_NAMESPACE_EXISTS')
 panel=ROOT/'audit/parallel_recovery_panel.json'
 cmd=[sys.executable,'-m','g1_pipeline.parallel_runner','run','--namespace',ns,'--panel',str(panel.relative_to(ROOT)),'--workers','1']
 log=(ROOT/'logs/parallel_boundary_reuse_test.log').open('ab')
 proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 wait_for(lambda:list((base/'items').glob('*/attempt_1/heartbeat.json')))
 stop=subprocess.run([sys.executable,'-m','g1_pipeline.parallel_runner','stop','--namespace',ns],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=30)
 assert stop.returncode==0
 proc.wait(timeout=180)
 summary=json.loads((base/'summary.json').read_text())
 assert summary['completed']==1 and summary['pending']==1 and summary['status']=='STOPPED_AT_CLIP_BOUNDARY'
 ledger=read_jsonl(base/'results.jsonl');first=ledger[0]
 first_hashes=dict(first['output_hashes'])
 resumed=subprocess.run(cmd+['--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240)
 assert resumed.returncode==0
 assert not (base/'STOP').exists() and list(base.glob('STOP_consumed_*.json'))
 summary=json.loads((base/'summary.json').read_text())
 assert summary['completed']==2 and summary['reused_clips']==1 and not summary['throughput_valid'] and summary['frames_per_sec'] is None
 assert compare(ns)['status']=='PASS'
 for p,h in first_hashes.items():assert sha(ROOT/p)==h
 # Independent namespace references old HML outputs, without invoking a solver.
 oldns=json.loads((ROOT/'audit/current_pilot_hml100_run.json').read_text())['namespace']
 old=read_jsonl(ROOT/'work'/oldns/'results.jsonl')
 mappings={r['sample_id']:r for r in json.loads((ROOT/'audit/hml_native_rest_mapping_v2.json').read_text())['results']}
 tasks=[{'source':'humanml3d','row':r,'mapping':mappings[r['sample_id']]} for r in read_jsonl(ROOT/'manifests/hml_smoke100.v2.jsonl')]
 ipanel=ROOT/'audit/parallel_hml100_reuse_panel.json';write_json(ipanel,{'scope':'verified import of completed authorized pilot; no additional attempts','tasks':tasks})
 ins='pilot_parallel_hml100_reuse_v2';ib=ROOT/'work'/ins
 before={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'gates/pilot/02_hml100.json',ROOT/'audit/current_pilot_hml100_run.json',ROOT/'work'/oldns/'results.jsonl']}
 imported=subprocess.run([sys.executable,'-m','g1_pipeline.parallel_runner','run','--namespace',ins,'--panel',str(ipanel.relative_to(ROOT)),'--workers','16'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240)
 assert imported.returncode==0
 current=read_jsonl(ib/'results.jsonl');s=json.loads((ib/'summary.json').read_text())
 assert current==old and s['completed']==100 and s['reused_clips']==100 and s['newly_processed_frames']==0
 assert not list((ib/'logs').glob('*.log'))
 for p,h in before.items():assert sha(ROOT/p)==h
 report={'status':'PASS','stop_namespace':ns,'clip_boundary_drained':True,'pending_not_started_until_resume':True,
         'resume_consumes_stop_request':True,'committed_clip_not_recomputed':True,'resume_not_reported_as_throughput':True,
         'exact_original_outputs':True,'reuse_namespace':ins,'old_hml_clips_reused':100,'new_hml_solver_attempts':0,
         'original_hml_gate_and_ledger_unchanged':before,'hml_final_quality_still':'87/100 retained,13 rejected,FAIL',
         'production_release':False}
 write_json(ROOT/'audit/parallel_boundary_reuse_test.json',report);print('BOUNDARY AND HML REUSE PASS')
 log.close()
if __name__=='__main__':run()
