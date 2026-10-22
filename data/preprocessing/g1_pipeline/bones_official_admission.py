"""2026-09-18 user-authorized BONES admission overlay; trajectories are immutable."""
import copy
from .common import ROOT,sha,write_json,digest
from .kinematic_policy import BASE,load

POLICY=BASE/'bones_official_20260918/policy.json'
REMOVED={'GROUND','GROUND_PENETRATION','SELF_COLLISION'}

def apply(entry,verify=True):
    if entry['dataset']!='bones':return entry
    policy=load(POLICY)
    if policy['scope']!='official_bones_only' or policy['geometry_admission']!='diagnostic_only':raise ValueError('INVALID_BONES_OFFICIAL_POLICY')
    if entry.get('bones_official_policy_sha256')==sha(POLICY):return entry
    old_reasons=entry.get('decision_reasons',entry.get('rejections',[]))
    # Restore ONLY exclusions whose complete cause was the withdrawn geometry gate.
    if entry.get('decision')!='REJECTED' or not old_reasons or not set(old_reasons)<=REMOVED:return entry
    if entry.get('gmr_role')!='official_g1_main_pool' or not entry.get('output_paths'):raise ValueError('BONES_OFFICIAL_OUTPUT_REQUIRED')
    if verify:
        meta=entry['source_metadata']
        if sha(meta['sealed_native_path'])!=entry['source_sha256']:raise ValueError('BONES_OFFICIAL_INPUT_CHANGED')
        for path,bound in entry['output_hashes'].items():
            if sha(ROOT/path)!=bound:raise ValueError('BONES_OFFICIAL_OUTPUT_CHANGED:'+path)
    contract=load(ROOT/entry['output_paths']['contract_path'])
    if contract['status']!='PASS' or not contract['finite']:raise ValueError('BONES_NUMERIC_CONTRACT_REQUIRED')
    result=copy.deepcopy(entry)
    history={'decision':entry['decision'],'reasons':old_reasons,'decision_path':entry['decision_path'],
             'decision_sha256':sha(ROOT/entry['decision_path']),'output_hashes':entry['output_hashes']}
    decision={'decision':'ACCEPTED_KINEMATIC','reasons':[],
              'diagnostic_tags':sorted(set(entry.get('diagnostic_tags',[]))|set(old_reasons)),
              'policy_path':str(POLICY.relative_to(ROOT)),'policy_sha256':sha(POLICY),
              'historical_policy_result':history,'trajectory_recomputed':False,
              'qualification_scope':'numeric official output; split/heldout/duplicate eligibility still required',
              'dynamic_validation':'NOT_RUN'}
    path=POLICY.parent/'decisions'/(digest(entry['sample_id'])+'.json')
    if path.exists():
        if load(path)!=decision:raise ValueError('OFFICIAL_REASSESSMENT_CONFLICT')
    else:write_json(path,decision)
    result.update(status='KINEMATIC_CANDIDATE',decision='ACCEPTED_KINEMATIC',decision_reasons=[],
                  decision_path=str(path.relative_to(ROOT)),diagnostic_tags=decision['diagnostic_tags'],
                  bones_official_policy_sha256=sha(POLICY),historical_policy_result=history,
                  admission_reused_output=True)
    result['output_hashes'][str(path.relative_to(ROOT))]=sha(path)
    return result

def execute(task,out,asset,expected_code):
    from .kinematic_clip import execute as original
    if task['source']!='bones':
        from . import source_motion_processing
        from .kinematic_20260918_runner import require_conversion_frozen
        source_motion_processing.require_frozen=require_conversion_frozen
    entry,execution=original(task,out,asset,expected_code)
    return apply(entry),execution
