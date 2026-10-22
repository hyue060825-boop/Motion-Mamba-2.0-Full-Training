"""Exercise the unchanged evaluator on imported pilot artifacts up to gate write."""
import json
import subprocess
import sys
from g1_pipeline.common import ROOT, read_jsonl, sha, write_json


class GateWriteIntercepted(Exception):
    pass


def run():
    ns = 'pilot_parallel_hml100_schema_v4'
    base = ROOT/'work'/ns
    assert not base.exists(), 'SCHEMA_TEST_NAMESPACE_EXISTS'
    oldns = json.loads((ROOT/'audit/current_pilot_hml100_run.json').read_text())['namespace']
    protected = [ROOT/'work'/oldns/'results.jsonl', ROOT/'audit/current_pilot_hml100_run.json']
    protected += list((ROOT/'gates').glob('*.json')) + list((ROOT/'gates/pilot').glob('*.json'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    with (ROOT/'logs/parallel_pilot_schema_test.log').open('ab') as log:
        result = subprocess.run([sys.executable, '-m', 'g1_pipeline.parallel_runner', 'run',
            '--namespace', ns, '--panel', 'audit/parallel_hml100_reuse_panel.json', '--workers', '16'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    assert result.returncode == 0
    summary = json.loads((base/'summary.json').read_text())
    assert summary['completed'] == summary['reused_clips'] == 100
    assert summary['newly_processed_frames'] == 0 and not list((base/'logs').glob('*.log'))
    assert not summary['throughput_valid'] and summary['frames_per_sec'] is None and summary['source_motion_hours_per_hour'] is None
    assert read_jsonl(base/'results.jsonl') == read_jsonl(ROOT/'work'/oldns/'results.jsonl')
    from g1_pipeline import source_smoke_evaluation as evaluator
    calls = []
    original_gate = evaluator.pilot_gate
    def intercept(*args, **kwargs):
        calls.append({'args': args, 'actual_quantity': kwargs['actual_quantity']})
        raise GateWriteIntercepted()
    evaluator.pilot_gate = intercept
    try:
        try:
            evaluator.evaluate(ns, 'humanml3d', pilot=True)
        except GateWriteIntercepted:
            pass
        else:
            raise AssertionError('ORIGINAL_EVALUATOR_DID_NOT_REACH_GATE')
    finally:
        evaluator.pilot_gate = original_gate
    assert len(calls) == 1 and calls[0]['actual_quantity'] == 100
    acceptance = ROOT/'qa/aggregate'/f'{ns}_acceptance.json'
    report = json.loads(acceptance.read_text())
    assert report['execution_scope'] == 'isolated_pilot' and report['attempted'] == 100
    assert report['status'] == 'FAIL' and not report['training_release']
    # No visuals/review were copied into the test namespace: it must not pass.
    assert 'SOURCE_SYSTEMATIC_REVIEW_MISSING' in report['failure_reasons']
    for path, bound in before.items():
        assert sha(ROOT/path) == bound, path
    write_json(ROOT/'audit/parallel_pilot_schema_test.json', {
        'status': 'PASS', 'namespace': ns, 'executor_sha256': sha(ROOT/'g1_pipeline/parallel_runner.py'),
        'original_evaluator_sha256': sha(ROOT/'g1_pipeline/source_smoke_evaluation.py'),
        'original_evaluator_reached_gate': True, 'gate_write_intercepted_only_in_test_process': True,
        'fixed100_artifacts_reused': 100, 'new_solver_attempts': 0,
        'unreviewed_test_pilot_remains_FAIL': True, 'original_protected_hashes': before,
        'test_acceptance_sha256': sha(acceptance), 'production_release': False})
    print('PILOT SCHEMA BRIDGE PASS; original gates unchanged')


if __name__ == '__main__':
    run()
