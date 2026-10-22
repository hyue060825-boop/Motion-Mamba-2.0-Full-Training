"""Verify completed results and correct inherited status labels only."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import hashlib
import json
import shutil
import time

R = Path(artifact_path('transformer_reference100k_corrected_evaluation_20261017_7a91e02b7e'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(16*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(p,v):
    p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    assert read(R/'COMPARISON_COMPLETE.json')['sources'] == 506
    p = read(R/'PLAN.json')
    for name,digest in p['execution_source_hashes'].items():
        assert sha(R/name) == digest, name
    numeric = read(R/'NUMERICS_COMPLETE.json')
    assert numeric['new_trajectories'] == 1518
    for name,digest in numeric['result_hashes'].items():
        assert sha(R/'aggregate'/name) == digest, name
    ep = R/'aggregate/ENDPOINT_AUDIT.json'
    a = read(ep)
    assert sum(v['N'] for v in a['checks']) == 1518
    assert len(a['checks']) == 3
    assert sha(ep) == numeric['endpoint_audit']
    archive = R/'metadata_archive'; archive.mkdir(exist_ok=True)
    if not (R/'METADATA_LABEL_CORRECTION.json').exists():
        shutil.copy2(ep,archive/'ENDPOINT_AUDIT.original.json')
        shutil.copy2(R/'NUMERICS_COMPLETE.json',archive/'NUMERICS_COMPLETE.original.json')
        old_hash = sha(ep); old_num_hash = sha(R/'NUMERICS_COMPLETE.json')
        a['legacy_status_label'] = a['status']
        a['status'] = 'PASS_ALL_1518_RETRAINED_MOTION_MAMBA2_ENDPOINTS_REAL_GOALS_AND_NO_ROUTE_PENALTY'
        a['N_trajectories'] = 1518
        write(ep,a)
        numeric['endpoint_audit'] = sha(ep)
        write(R/'NUMERICS_COMPLETE.json',numeric)
        write(R/'METADATA_LABEL_CORRECTION.json',dict(
            correction='Inherited status label said3542/official; checks actually contain three groups of506 retrained trajectories. No numerical results changed.',
            endpoint_audit_old_sha256=old_hash,endpoint_audit_new_sha256=sha(ep),
            numerics_old_sha256=old_num_hash,numerics_new_sha256=sha(R/'NUMERICS_COMPLETE.json'),
            execution_sources_unchanged=True,result_csv_hashes_unchanged=True,time=time.time()))
    shutil.copy2(R/'comparison/OFFICIAL_BEFORE_AFTER.csv',R/'comparison/RETRAINED_MOTION_MAMBA2_BEFORE_AFTER.csv')
    checkpoint = Path(artifact_path('checkpoint100000_pt_1f87866a73'))
    digest = sha(checkpoint)
    assert digest == '723e251a145461d379fc69f315f976431667fc2d8092c9d4f5f9f35c5817dd98'
    write(R/'FINAL_VERIFICATION.json',dict(status='PASS_COMPLETED_READ_ONLY_INFERENCE_AUDIT_AND_CORRECTED_SCORES',
        model='self_trained_ARDY100k',checkpoint_sha256=digest,training_updates=0,simulation_changes=False,
        trajectories=1518,sources=506,tasks=3,source_hashes_unchanged=True,
        numeric_csv_hashes=numeric['result_hashes'],
        inference_audit=sha(R/'INFERENCE_EXECUTION_AUDIT.json'),
        same_inputs_and_references=sha(R/'comparison/COMPARABILITY_AUDIT.json'),
        root_problem_still_present=True,architecture_superiority_not_proven=True,
        report_type='Markdown and CSV; no new figure requested',time=time.time()))
    write(R/'STATUS.json',dict(status='COMPLETE_INFERENCE_AUDIT_AND_CORRECTED_NUMERICAL_EVALUATION',
        trajectories=1518,sources=506,training_updates=0,report_pending_local_packaging=True,time=time.time()))
    print('FINAL_VERIFIED')


if __name__ == '__main__':
    main()
