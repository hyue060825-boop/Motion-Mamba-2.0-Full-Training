"""Exact numeric and QA comparison of full clips against original serial evidence."""
import argparse
import json
import numpy as np
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha


def compare(namespace):
 base=ROOT/'work'/namespace
 design=json.loads((base/'run.json').read_text())
 tasks=json.loads((ROOT/design['panel']).read_text())['tasks']
 records={r['sample_id']:r for r in read_jsonl(base/'results.jsonl')}
 results=[]
 for task in tasks:
  sid=task['row']['sample_id'];ref=task['reference'];candidate=records.get(sid)
  if candidate is None:continue
  issues=[];checks={}
  if sha(ROOT/task['reference_run_path'])!=task['reference_run_sha256']:raise ValueError('REFERENCE_DESIGN_CHANGED')
  for path,bound in ref['output_hashes'].items():
   if sha(ROOT/path)!=bound:raise ValueError('REFERENCE_OUTPUT_CHANGED:'+path)
  for path,bound in candidate.get('output_hashes',{}).items():
   if sha(ROOT/path)!=bound:raise ValueError('CANDIDATE_OUTPUT_CHANGED:'+path)
  for key in ('status','frames','rejections','review_flags','contract_status','source_semantics','mapping_sha256','source_sha256'):
   checks[key]=candidate.get(key)==ref.get(key)
  if candidate.get('output_paths'):
   ra=(ROOT/ref['output_paths']['qpos_path']).parent;ca=(ROOT/candidate['output_paths']['qpos_path']).parent
   for name in ('qpos.npy','unprocessed_qpos.npy','features.npy'):
    a=np.load(ra/name);b=np.load(ca/name)
    same_shape=a.shape==b.shape
    checks[name]={'exact':same_shape and bool(np.array_equal(a,b)),
                  'max_abs':float(np.max(np.abs(a-b))) if same_shape else None,'same_dtype':a.dtype==b.dtype}
   with np.load(ra/'human_targets.npz') as a,np.load(ca/'human_targets.npz') as b:
    checks['targets_members']=set(a.files)==set(b.files)
    for key in a.files:
     checks['human_targets:'+key]={'exact':key in b and np.array_equal(a[key],b[key]),
                                  'max_abs':float(np.max(np.abs(a[key]-b[key]))) if key in b and a[key].shape==b[key].shape else None}
   for name in ('time.json','solver.json','contract.json'):
    checks[name]=json.loads((ra/name).read_text())==json.loads((ca/name).read_text())
   checks['csv_bytes']=sha(ra/'motion.csv')==sha(ca/'motion.csv')
   checks['sidecar_content']=json.loads((ra/'motion.json').read_text())==json.loads((ca/'motion.json').read_text())
   # Only absolute feature location differs in the actual loader input manifest.
   a=read_jsonl(ra/'project_loader.jsonl');b=read_jsonl(ca/'project_loader.jsonl')
   checks['loader_manifest_except_output_location']=[{k:v for k,v in r.items() if k!='feature_path'} for r in a]==[{k:v for k,v in r.items() if k!='feature_path'} for r in b]
  for key,value in checks.items():
   passed=value.get('exact',False) and value.get('same_dtype',True) if isinstance(value,dict) else value
   if not passed:issues.append(key)
  from parallel_roundoff_check import check as roundoff_check
  roundoff=roundoff_check(ra,ca,issues) if candidate.get('output_paths') else None
  results.append({'sample_id':sid,'status':'PASS' if not issues or roundoff else 'FAIL',
                  'issues':[] if roundoff else issues,'strict_equality_differences':issues,
                  'historical_roundoff':roundoff,'checks':checks})
 report={'status':'PASS' if len(results)==len(tasks) and all(r['status']=='PASS' for r in results) else 'INCOMPLETE' if len(results)<len(tasks) else 'FAIL',
         'namespace':namespace,'expected':len(tasks),'compared':len(results),'results':results,
         'tolerance':'Exact preferred. Diagnosed historical postprocessing-QP rounding uses existing frozen numeric contract only; CSV/features/time/contracts/decisions and preprocessed arrays must stay exact; differing derived scalars independently verified.',
         'excluded':'output directory references, execution timing, pid, attempt and scheduler logs only',
         'bindings':{str(base.relative_to(ROOT)/'run.json'):sha(base/'run.json'),str(base.relative_to(ROOT)/'results.jsonl'):sha(base/'results.jsonl')}}
 write_json(ROOT/f'audit/{namespace}_comparison.json',report)
 print({k:v for k,v in report.items() if k not in ('results','bindings')})
 for r in results:
  if r['issues']:print(r['sample_id'],r['issues'])
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--namespace',required=True);a=p.parse_args();compare(a.namespace)
