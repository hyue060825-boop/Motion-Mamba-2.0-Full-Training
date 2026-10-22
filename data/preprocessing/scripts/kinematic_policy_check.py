"""Check only the new admission branch against actual frozen pilot outcomes."""
import copy,json
from g1_pipeline.common import ROOT,sha,write_json,read_jsonl
from g1_pipeline.kinematic_policy import BASE,load,decide

def run():
 before={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'gates').glob('*.json'))+list((ROOT/'gates/pilot').glob('*.json'))+[ROOT/'release_manifest.json',ROOT/'audit/current_work_state.json']}
 retained=[];hard=[];outcomes=[]
 for short in ['hml','hiphi']:
  ns=load(ROOT/f'audit/current_pilot_{short}100_run.json')['namespace']
  decisions={r['sample_id']:r for r in load(ROOT/f'audit/{ns}_review_decisions.json')['decisions']}
  for entry in read_jsonl(ROOT/'work'/ns/'results.jsonl'):
   metrics=load(ROOT/entry['metrics_path']);contract=load(ROOT/entry['output_paths']['contract_path'])
   prior={'entry':entry,'decision':decisions[entry['sample_id']]}
   result=decide(entry,metrics,contract,prior=prior)
   if entry['rejections']:
    original_hard=set(entry['rejections'])-{'RELATIVE_ACTION_CLASS_UNSUPPORTED','RELATIVE_CATEGORY_UNSUPPORTED','EXTERNAL_OR_UNRESOLVED_SUPPORT_SCENE'}
    if original_hard:assert result['decision']=='REJECTED';hard.append(entry['sample_id'])
   if result['decision']=='ACCEPTED_KINEMATIC':retained.append(entry['sample_id'])
   if not entry['rejections'] and not retained:pass
   outcomes.append({'sample_id':entry['sample_id'],**result})
 # Actual clean retained record: diagnostic relative tags/dynamic NOT_RUN do not block.
 sid=retained[0]
 ns=load(ROOT/'audit/current_pilot_hml100_run.json')['namespace']
 good=next(r for r in read_jsonl(ROOT/'work'/ns/'results.jsonl') if r['sample_id']==sid)
 m=load(ROOT/good['metrics_path']);c=load(ROOT/good['output_paths']['contract_path'])
 fake=copy.deepcopy(good);fake['rejections']=['RELATIVE_ACTION_CLASS_UNSUPPORTED'];fake['review_flags']=[]
 baseline=decide(fake,m,c)
 assert baseline['decision']=='ACCEPTED_KINEMATIC' and baseline['dynamic_validation']=='NOT_RUN'
 for code in ['RATE_CORRECTION_EXCEEDS_CONTENT_CHANGE_BUDGET','SELF_COLLISION','GROUND_PENETRATION','NUMERICAL_CONTRACT_FAILURE']:
  bad=copy.deepcopy(fake);bad['rejections'].append(code)
  assert decide(bad,m,c)['decision']=='REJECTED',code
 for p,b in before.items():assert sha(ROOT/p)==b,p
 result={'status':'PASS','scope':'new offline admission only','tested_original_clips':len(outcomes),'old_hard_failure_ids_still_rejected':hard,'accepted_with_dynamic_not_run':len(retained),'low_aggregate_yield_not_a_gate':True,'diagnostic_category_only_not_rejected':True,'hard_failure_negative_cases':4,'original_gates_and_release_unchanged':before,'results':outcomes,'policy_sha256':sha(BASE/'execution_policy.json'),'policy_module_sha256':sha(ROOT/'g1_pipeline/kinematic_policy.py')}
 write_json(BASE/'policy_branch_checks.json',result);print({k:result[k] for k in ['status','tested_original_clips','accepted_with_dynamic_not_run','hard_failure_negative_cases']})
if __name__=='__main__':run()
