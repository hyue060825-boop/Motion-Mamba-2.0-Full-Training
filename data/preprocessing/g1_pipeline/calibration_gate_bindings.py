"""Bind a passed calibration gate to its frozen execution, not later source code.

This may only refine an already valid PASS. It preserves all acceptance metrics
and adds direct hashes for source motions and generated evidence. No threshold,
trajectory, or acceptance disposition is changed.
"""
import json
from .common import ROOT,BONES,read_jsonl,sha,digest,write_json,atomic_bytes,now
from .gates import require_pass,fingerprint
from .calibration_protocol import require_frozen,PATH


def bind(namespace='bones_frozen_accept_v25'):
    state=require_pass('01_bones_calibration');policy=require_frozen()
    if state.get('binding_revision',{}).get('policy_sha256')==sha(PATH):return state
    report_path=ROOT/'qa/aggregate'/f'{namespace}.json';run=json.loads(report_path.read_text())
    if run['run_config']['policy_sha256']!=sha(PATH) or run['run_config']['panel']!='accept':raise ValueError('WRONG_ACCEPTANCE_EXECUTION')
    rows={r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_accept.jsonl')}
    inputs=dict(state['input_hashes']);inputs.update(policy['bindings'])
    inputs[str(PATH.relative_to(ROOT))]=sha(PATH)
    for r in run['results']:
        row=rows[r['sample_id']];mid=row['source_motion_id'];base=ROOT/'work'/namespace
        files=[(base/(mid+'.npy'),'output_sha256'),(base/(mid+'.time.json'),'time_sha256'),
               (base/(mid+'.solver.json'),'solver_sha256'),(base/(mid+'.unprocessed.npy'),'unprocessed_sha256'),
               (ROOT/'work/bones_pairs_accept'/(mid+'.bvh'),'source_sha256'),(BONES/row['source_paths']['g1'],'reference_sha256')]
        for p,key in files:
            if sha(p)!=r[key]:raise ValueError('ACCEPTANCE_ARTIFACT_CHANGED:'+str(p))
            inputs[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=r[key]
    target=json.loads((ROOT/'qa/aggregate'/f'{namespace}_human_targets.json').read_text())
    for r in target['results']:
        from pathlib import Path
        p=Path(r['per_frame_path'])
        if sha(p)!=r['per_frame_sha256']:raise ValueError('ACCEPTANCE_TARGET_EVIDENCE_CHANGED')
        inputs[str(p.relative_to(ROOT))]=r['per_frame_sha256']
    reviewpath=ROOT/'audit'/f'{namespace}_review_decisions.json'
    if reviewpath.exists():
        review=json.loads(reviewpath.read_text())
        if review['policy_sha256']!=sha(PATH):raise ValueError('ACCEPTANCE_REVIEW_POLICY_CHANGED')
        for name,bound in review['bindings'].items():
            if sha(ROOT/name)!=bound:raise ValueError('ACCEPTANCE_VISUAL_EVIDENCE_CHANGED:'+name)
            inputs[name]=bound
    own='g1_pipeline/calibration_gate_bindings.py'
    scope={'code_paths':sorted([p for p in policy['bindings'] if p.endswith('.py')]+[own]),
           'config_paths':sorted(p for p in policy['bindings'] if p.startswith('configs/'))}
    # The bound policy itself records the authoritative pre-held-out closure.
    scope['config_paths'].append(str(PATH.relative_to(ROOT)))
    path=ROOT/'gates/01_bones_calibration.json';previous=sha(path)
    atomic_bytes(ROOT/'audit/gate_binding_history'/(previous+'.json'),path.read_bytes())
    state.update(**fingerprint(scope),fingerprint_scope=scope,input_hashes=inputs,input_manifest_hash=digest(inputs),
        binding_revision={'time':now(),'previous_gate_sha256':previous,'policy_sha256':sha(PATH),
            'reason':'Bind verified frozen calibration dependencies and materialized artifacts; later independent source modules cannot invalidate unchanged upstream semantics.'})
    write_json(path,state);return require_pass('01_bones_calibration')


if __name__=='__main__':print(bind()['status'])
