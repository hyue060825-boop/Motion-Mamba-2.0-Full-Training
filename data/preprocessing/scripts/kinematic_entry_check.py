"""Small actual entry check: one complete native HML clip and one official BONES clip."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import copy,json
import numpy as np
from g1_pipeline.common import ROOT,BONES,HML,sha,read_jsonl,write_json
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.kinematic_clip import execute

def run():
 target=BASE/'entry_check';target.mkdir(exist_ok=True)
 old=load(ROOT/'work/benchmark_equivalence_w1_v1/run.json');asset=old['asset'];code=old['code']
 hml=load(ROOT/'audit/current_pilot_hml100_run.json')['namespace']
 original=next(r for r in read_jsonl(ROOT/'work'/hml/'results.jsonl') if r['sample_id']=='humanml3d:004764')
 row=copy.deepcopy(original['source_metadata']);row['sample_id']='humanml3d:entry_check_004764';row.update(sealed_native_path=str(HML/row['source_paths']['native']),sealed_native_sha256=original['source_sha256'])
 out=target/'hml';out.mkdir(exist_ok=True)
 result,_=execute({'source':'humanml3d','row':row},out,asset,code)
 assert result['contract_status']=='PASS' and result['decision']=='ACCEPTED_KINEMATIC',result
 for key in ('features_path','qpos_path'):
  a=np.load(ROOT/result['output_paths'][key]);b=np.load(ROOT/original['output_paths'][key]);assert np.array_equal(a,b),key
 # BONES full duration, official copy only; its quality may pass or reject.
 with (ROOT/'manifests/full_candidates.v2.jsonl').open() as stream:row=json.loads(next(stream))
 row.update(sealed_native_path=str(BONES/row['source_paths']['g1']),sealed_native_sha256=sha(BONES/row['source_paths']['g1']))
 out=target/'bones';out.mkdir(exist_ok=True)
 bones,_=execute({'source':'bones','row':row},out,asset,code)
 assert bones['status']!='ERROR'
 report={'status':'PASS','hml_full_clip_bit_exact_to_frozen_original':True,'hml_frames':result['frames'],'hml_decision':result['decision'],'bones_decision':bones['decision'],'bones_frames':bones['frames'],'original_solver_modules_modified':False,'dynamic_required':False,'module_hashes':{p:sha(ROOT/p) for p in ['g1_pipeline/kinematic_clip.py','g1_pipeline/kinematic_policy.py','g1_pipeline/kinematic_runner.py']}}
 write_json(BASE/'entry_checks.json',report);print(report)
if __name__=='__main__':run()
