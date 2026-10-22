"""Check a complete staged HiPHI clip through the new scope and original v25 body."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json,copy
import numpy as np
from g1_pipeline.common import ROOT,sha,read_jsonl,write_json
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.kinematic_clip import execute
old=load(ROOT/'work/benchmark_equivalence_w1_v1/run.json')
ns=load(ROOT/'audit/current_pilot_hiphi100_run.json')['namespace']
rows=read_jsonl(ROOT/'work'/ns/'results.jsonl')
packages={r['sample_id']:r for p in (BASE/'inputs/hiphi').glob('*.index.json') for r in load(p)['results']}
original=min((r for r in rows if r['sample_id'] in packages),key=lambda r:r['frames'])
row=copy.deepcopy(original['source_metadata']);row['sample_id']='hiphi:entry_check_'+row['source_motion_id'];row.update(sealed_native_sha256=original['source_sha256'],package=packages[original['sample_id']])
out=BASE/'entry_check/hiphi';out.mkdir(parents=True,exist_ok=False)
entry,execution=execute({'source':'hiphi','row':row},out,old['asset'],old['code'])
assert entry['contract_status']=='PASS'
comparisons={}
for key in ('qpos_path','features_path'):
 a=np.load(ROOT/entry['output_paths'][key]);b=np.load(ROOT/original['output_paths'][key]);comparisons[key]={'exact':bool(np.array_equal(a,b)),'max_abs':float(np.max(np.abs(a-b)))}
 assert comparisons[key]['exact'],comparisons
result={'status':'PASS','source_sample_id':original['sample_id'],'complete_frames':entry['frames'],'decision':entry['decision'],'comparison':comparisons,'native_compressed_input_verified':True,'module_sha256':sha(ROOT/'g1_pipeline/kinematic_clip.py'),'policy_sha256':sha(BASE/'execution_policy.json'),'scope':'new entry check only; no fixed100 reevaluation or benchmark'}
write_json(BASE/'hiphi_entry_checks.json',result);print(result)
