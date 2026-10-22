"""Freeze representative full-clip benchmark panels from the existing pilots."""
import json
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha,digest

def run():
 tasks={};references={}
 for short,source,manifest,static in [('hml','humanml3d','hml_smoke100.v2.jsonl','hml_native_rest_mapping_v2.json'),('hiphi','hiphi','hiphi_smoke100.jsonl','hiphi_native_rest_mapping_v1.json')]:
  ns=json.loads((ROOT/f'audit/current_pilot_{short}100_run.json').read_text())['namespace']
  records={r['sample_id']:r for r in read_jsonl(ROOT/'work'/ns/'results.jsonl')}
  mappings={r['sample_id']:r for r in json.loads((ROOT/'audit'/static).read_text())['results']}
  packages={r['sample_id']:r for r in json.loads((ROOT/'audit/hiphi_package_evidence_v1.json').read_text())['results']}
  for row in read_jsonl(ROOT/'manifests'/manifest):
   sid=row['sample_id']
   if sid not in records:continue
   record=records[sid]
   for path,bound in record.get('output_hashes',{}).items():
    if sha(ROOT/path)!=bound:raise ValueError('OLD_OUTPUT_CHANGED:'+path)
   task={'source':source,'row':row,'mapping':mappings[sid], 'reference':record,
         'reference_run_path':f'work/{ns}/run.json','reference_run_sha256':sha(ROOT/'work'/ns/'run.json')}
   if short=='hiphi':task['package']=packages[sid]
   tasks[sid]=task
 ids=['humanml3d:004764','humanml3d:002085','hiphi:Cause_motion-hit_0042','hiphi:Patrolling-patrol_0032']
 small=[tasks[s] for s in ids]
 # Same fixed 64 full clips at 16 and 32 workers: 32 from each source,
 # include ordinary, hard reject, and the long complete HiPHI clip.
 selected=list(ids)
 for source in ['humanml3d','hiphi']:
  pool=sorted([s for s,t in tasks.items() if t['source']==source],key=lambda s:digest([20260915,s]))
  for sid in pool:
   if sum(tasks[s]['source']==source for s in selected)>=32:break
   if sid not in selected:selected.append(sid)
 for name,group in [('equivalence',small),('throughput',[tasks[s] for s in selected])]:
  path=ROOT/f'audit/parallel_{name}_panel.json'
  obj={'seed':20260915,'scope':'isolated benchmark; complete original clips; not training count',
       'selection':'fixed deterministic panel including explicit ordinary, high-dynamic rejection and long-clip sentinels',
       'tasks':group,'frames':sum(t['reference']['frames'] for t in group)}
  if path.exists():
   old=json.loads(path.read_text())
   if old!=obj:raise ValueError('BENCHMARK_PANEL_ALREADY_FROZEN')
  else:write_json(path,obj)
  print(name,len(group),obj['frames'])
if __name__=='__main__':run()
