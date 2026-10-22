"""Small guard checks: no source mutations and no production promotion."""
import copy
import os
import json
import subprocess
import sys
from g1_pipeline.common import ROOT,sha,write_json,read_jsonl

def run():
 baseline=json.loads((ROOT/'audit/pilot_finalization_production_baseline.json').read_text())['hashes']
 for path,bound in baseline.items():assert sha(ROOT/path)==bound
 template=json.loads((ROOT/'audit/parallel_equivalence_panel.json').read_text())
 task=copy.deepcopy(template['tasks'][0]);task['mapping']['native_sha256']='0'*64
 obj={'scope':'isolated incorrect-hash negative test; no source changed','tasks':[task]}
 panel=ROOT/'audit/parallel_bad_hash_panel.json';write_json(panel,obj)
 ns='benchmark_bad_hash_'+os.environ.get('PARALLEL_VALIDATION_VERSION','v1')
 proc=subprocess.run([sys.executable,'-m','g1_pipeline.parallel_runner','run','--namespace',ns,'--panel',str(panel.relative_to(ROOT)),'--workers','1'],cwd=ROOT,capture_output=True,text=True,timeout=120)
 rows=read_jsonl(ROOT/'work'/ns/'results.jsonl')
 assert len(rows)==1 and rows[0]['status']=='ERROR' and 'SEALED_HML_INPUT_CHANGED' in rows[0]['error']
 assert not rows[0].get('output_paths')
 from g1_pipeline.production_hml import run as hml
 from g1_pipeline.production_hiphi import run as hiphi
 guards=[]
 for source,fn in [('humanml3d',hml),('hiphi',hiphi)]:
  try:fn(dry_run=True)
  except Exception as exc:
   assert 'GATE_' in str(exc);guards.append({'source':source,'error':str(exc)})
  else:raise AssertionError('PRODUCTION_DEFAULT_WAS_ALLOWED')
 for path,bound in baseline.items():assert sha(ROOT/path)==bound
 write_json(ROOT/'audit/parallel_guard_tests.json',{'status':'PASS','bad_native_hash_rejected':True,
  'bad_hash_error':rows[0]['error'],'bad_hash_output_paths_absent':True,'production_default_guards':guards,
  'production_hashes_unchanged':baseline,'test_input_only':'isolated task mapping copy changed; original native/config/manifests untouched'})
 print('GUARDS PASS')
if __name__=='__main__':run()
