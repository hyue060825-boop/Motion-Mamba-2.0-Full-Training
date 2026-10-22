"""Official BONES admission at manifest publication, including ground-scope diagnostics.

The running converter and its sealed bindings remain immutable. Its raw decisions
are historical policy results; this layer is authoritative for training admission.
"""
import copy
from .common import ROOT, sha, write_json, digest
from .kinematic_policy import load
from .bones_official_admission import POLICY, apply as geometry_admission

ADDENDUM = POLICY.parent / 'scene_scope_policy.json'


def apply(entry, verify=True):
    result = geometry_admission(entry, verify=verify)
    if result['dataset'] != 'bones':
        return result
    reasons = result.get('decision_reasons', [])
    # This reason originates exclusively in the ground/contact applicability
    # branch of kinematic_policy.decide, not in source fidelity or the contract.
    if result.get('decision') != 'REVIEW_PENDING' or set(reasons) != {'UNRESOLVED_APPLICABLE_SCENE'}:
        return result
    policy = load(ADDENDUM)
    if policy['parent_policy_sha256'] != sha(POLICY) or policy['withdrawn_pending_reason'] != reasons[0]:
        raise ValueError('BONES_SCENE_POLICY_MISMATCH')
    if sha(ROOT/'g1_pipeline/kinematic_policy.py') != policy['reason_origin_sha256']:
        raise ValueError('BONES_SCENE_REASON_ORIGIN_CHANGED')
    if result.get('gmr_role') != 'official_g1_main_pool' or not result.get('output_paths'):
        raise ValueError('BONES_OFFICIAL_OUTPUT_REQUIRED')
    if verify:
        if sha(result['source_metadata']['sealed_native_path']) != result['source_sha256']:
            raise ValueError('BONES_OFFICIAL_INPUT_CHANGED')
        for path, bound in result['output_hashes'].items():
            if sha(ROOT/path) != bound:
                raise ValueError('BONES_OFFICIAL_OUTPUT_CHANGED:'+path)
    contract = load(ROOT/result['output_paths']['contract_path'])
    if contract['status'] != 'PASS' or not contract['finite'] or contract['hinge_limit_excess_max_rad'] > 1e-4:
        raise ValueError('BONES_NUMERIC_CONTRACT_REQUIRED')
    result = copy.deepcopy(result)
    history = {'decision': result['decision'], 'reasons': reasons,
               'decision_path': result['decision_path'], 'decision_sha256': sha(ROOT/result['decision_path']),
               'output_hashes': copy.deepcopy(result['output_hashes'])}
    decision = {'decision': 'ACCEPTED_KINEMATIC', 'reasons': [],
                'diagnostic_tags': sorted(set(result.get('diagnostic_tags', [])) | set(reasons)),
                'policy_sha256': sha(POLICY), 'scene_scope_policy_sha256': sha(ADDENDUM),
                'historical_policy_result': history, 'trajectory_recomputed': False,
                'qualification_scope': 'official numeric output; split/heldout/duplicate eligibility still required',
                'dynamic_validation': 'NOT_RUN'}
    path = POLICY.parent/'scene_decisions'/(digest(result['sample_id'])+'.json')
    if path.exists():
        if load(path) != decision:
            raise ValueError('BONES_SCENE_REASSESSMENT_CONFLICT')
    else:
        write_json(path, decision)
    result.update(status='KINEMATIC_CANDIDATE', decision='ACCEPTED_KINEMATIC', decision_reasons=[],
                  decision_path=str(path.relative_to(ROOT)), diagnostic_tags=decision['diagnostic_tags'],
                  bones_official_policy_sha256=sha(POLICY), scene_scope_policy_sha256=sha(ADDENDUM),
                  historical_policy_result=history, admission_reused_output=True)
    result['output_hashes'][str(path.relative_to(ROOT))] = sha(path)
    return result
