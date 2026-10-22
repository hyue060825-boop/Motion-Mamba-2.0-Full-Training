"""Verify completed pilot evidence without promoting or rewriting production state."""
import json
from g1_pipeline.common import ROOT, read_jsonl, sha, digest, write_json, now
from g1_pipeline.pilot_scope import require_pilot
from g1_pipeline.gates import fingerprint


def load(path):
    return json.loads((ROOT/path).read_text())


def verify(bindings):
    failures=[]
    for path, expected in bindings.items():
        p=ROOT/path
        if not p.is_file() or sha(p)!=expected:
            failures.append(path)
    if failures:
        raise ValueError('EVIDENCE_CHANGED:'+repr(failures))
    return len(bindings)


def run():
    require_pilot('humanml3d');require_pilot('hiphi')
    frozen=load('audit/bones_frozen_accept_v25_final_evidence_audit.json')
    checked=verify(frozen['checked_hashes'])
    baseline=load('audit/pilot_finalization_production_baseline.json')
    verify(baseline['hashes'])
    release=load('release_manifest.json')
    verify({r['path']:r['sha256'] for r in release['gates'].values()})
    assert release['accepted_counts']=={'bones':0,'humanml3d':0,'hiphi':0}
    assert not release['training_data_ready']
    sources={}
    for short,stage in [('hml','02_hml100'),('hiphi','03_hiphi100')]:
        ns=load(f'audit/current_pilot_{short}100_run.json')['namespace']
        state=load(f'gates/pilot/{stage}.json')
        report=load(f'qa/aggregate/{ns}_acceptance.json')
        review=load(f'audit/{ns}_review_decisions.json')
        rows=read_jsonl(ROOT/'work'/ns/'results.jsonl')
        assert state['execution_scope']=='isolated_pilot' and not state['training_release']
        assert state['actual_quantity']==100 and state['status']==report['status']
        assert all(state[k]==v for k,v in fingerprint(state['fingerprint_scope']).items())
        verify(state['input_hashes']);verify(review['bindings'])
        assert state['input_manifest_hash']==digest(state['input_hashes'])
        assert len(rows)==len({r['sample_id'] for r in rows})==100
        assert {r['sample_id'] for r in rows}=={r['sample_id'] for r in review['decisions']}
        assert len(review['decisions'])==100 and not report['review_pending_ids']
        accepted=read_jsonl(ROOT/'work'/ns/'accepted.jsonl')
        rejected=read_jsonl(ROOT/'work'/ns/'rejected.jsonl')
        assert len(accepted)==report['kinematic_retained'] and len(accepted)+len(rejected)==100
        assert all(not r['training_release'] for r in accepted)
        sources[short]={'namespace':ns,'attempted':100,'accepted':len(accepted),
            'rejected':len(rejected),'review_pending':0,'status':report['status'],
            'gate_sha256':sha(ROOT/f'gates/pilot/{stage}.json'),
            'gate_bound_files':len(state['input_hashes']),'all_review_bindings_verified':True}
    result={'time':now(),'status':'EVIDENCE_CONSISTENT','scope':'Completed isolated pilots only',
        'bones_original_files_verified':checked,'mismatches':[],
        'production_gates_verified_against_original_release_manifest':7,
        'production_checkpoint_hashes_verified':baseline['hashes'],
        'original_bones_status':'FAIL','original_core_input':118,'original_core_retained':112,
        'policy_sha256':frozen['policy_sha256'],'sources':sources,
        'production_accepted':release['accepted_counts'],'production_release':False,
        'training_data_ready':False}
    write_json(ROOT/'audit/pilot_final_evidence_audit.json',result)
    print({k:result[k] for k in ['status','bones_original_files_verified','sources','production_accepted']})


if __name__=='__main__':
    run()
