"""Finish the parallel handoff by referencing the naturally completed serial pilot."""
import json
import subprocess
import sys
from g1_pipeline.common import ROOT, read_jsonl, sha, write_json


def run():
    upgrade = json.loads((ROOT/'audit/parallel_retry_executor_upgrade.json').read_text())
    assert upgrade['status'] == 'VERIFIED'
    assert upgrade['to_sha256'] == sha(ROOT/'g1_pipeline/parallel_runner.py')
    selection = json.loads((ROOT/'audit/parallel_worker_selection.json').read_text())
    workers = selection['selected_workers']
    oldns = json.loads((ROOT/'audit/current_pilot_hiphi100_run.json').read_text())['namespace']
    oldbase = ROOT/'work'/oldns
    original = read_jsonl(oldbase/'results.jsonl')
    assert len(original) == len({r['sample_id'] for r in original}) == 100
    assert not json.loads((ROOT/'qa/aggregate'/f'{oldns}_acceptance.json').read_text())['review_pending_ids']
    mapping = {r['sample_id']: r for r in json.loads((ROOT/'audit/hiphi_native_rest_mapping_v1.json').read_text())['results']}
    packages = {r['sample_id']: r for r in json.loads((ROOT/'audit/hiphi_package_evidence_v1.json').read_text())['results']}
    panel = ROOT/'audit/parallel_hiphi100_reuse_panel.json'
    write_json(panel, {'scope': 'Original fixed100 pilot outputs, reference only, no new solver attempts',
        'tasks': [{'source': 'hiphi', 'row': r, 'mapping': mapping[r['sample_id']], 'package': packages[r['sample_id']]}
                  for r in read_jsonl(ROOT/'manifests/hiphi_smoke100.jsonl')]})
    protected = [oldbase/'results.jsonl', oldbase/'run.json', oldbase/'accepted.jsonl', oldbase/'rejected.jsonl',
                 ROOT/'gates/pilot/03_hiphi100.json', ROOT/'gates/pilot/02_hml100.json', ROOT/'release_manifest.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    ns = 'pilot_parallel_hiphi100_reuse_v4'
    base = ROOT/'work'/ns
    assert not base.exists(), 'HIPHI_REUSE_NAMESPACE_EXISTS'
    with (ROOT/'logs/parallel_hiphi_reuse.log').open('ab') as log:
        result = subprocess.run([sys.executable, '-m', 'g1_pipeline.parallel_runner', 'run',
            '--namespace', ns, '--panel', str(panel.relative_to(ROOT)), '--workers', str(workers)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    assert result.returncode == 0
    summary = json.loads((base/'summary.json').read_text())
    assert summary['completed'] == summary['reused_clips'] == 100
    assert summary['newly_processed_frames'] == 0 and not summary['throughput_valid']
    assert not list((base/'logs').glob('*.log'))
    assert read_jsonl(base/'results.jsonl') == original
    for path, bound in before.items():
        assert sha(ROOT/path) == bound, path
    write_json(ROOT/'audit/parallel_hiphi_reuse.json', {
        'status': 'PASS', 'namespace': ns, 'selected_workers': workers,
        'executor_sha256': sha(ROOT/'g1_pipeline/parallel_runner.py'),
        'original_namespace': oldns, 'reused_clips': 100, 'new_solver_attempts': 0,
        'original_gate_ledger_acceptance_unchanged': before,
        'parallel_summary_sha256': sha(base/'summary.json'), 'production_release': False})
    print('HiPHI completed pilot handoff verified: 100 reused, zero solver reruns')


if __name__ == '__main__':
    run()
