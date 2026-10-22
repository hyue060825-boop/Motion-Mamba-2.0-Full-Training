"""Pilot authorization must never relax production admission."""
import json
import pytest
from g1_pipeline import pilot_scope, production_hml, production_hiphi
from g1_pipeline import source_smoke_evaluation as evaluation
from g1_pipeline.common import sha


def test_production_defaults_still_require_production_gate():
    with pytest.raises(RuntimeError, match='GATE_NOT_PASS:01_bones_calibration:FAIL'):
        production_hml.run(dry_run=True)
    with pytest.raises(RuntimeError, match='GATE_NOT_PASS:02_hml100:BLOCKED'):
        production_hiphi.run(dry_run=True)


def test_pilot_status_writer_cannot_touch_production_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(pilot_scope, 'gate', lambda *a, **k: calls.append((a, k)))
    pilot_scope.pilot_gate('02_hml100', 'PASS', actual_quantity=100,
                           upstream_dependencies=['01_bones_calibration'])
    args, kwargs = calls[0]
    assert args[0] == 'pilot/02_hml100'
    assert kwargs['execution_scope'] == 'isolated_pilot'
    assert kwargs['training_release'] is False
    assert kwargs['upstream_dependencies'] == []
    with pytest.raises(ValueError, match='INVALID_PILOT_STAGE'):
        pilot_scope.pilot_gate('01_bones_calibration', 'PASS')


def test_production_evaluator_rejects_pilot_namespace_even_if_gate_passes(monkeypatch):
    monkeypatch.setattr(evaluation, 'require_pass', lambda stage: None)
    with pytest.raises(ValueError, match='SOURCE_EXECUTION_SCOPE_MISMATCH'):
        evaluation.evaluate('pilot_hml100_test', 'humanml3d')


def test_pilot_authorization_fails_on_changed_protected_evidence(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('{}\n')
    protected = tmp_path / 'frozen_evaluation.json'
    protected.write_text('{"status":"FAIL"}')
    policy = tmp_path / 'policy.json'
    policy.write_text('{}')
    auth = tmp_path / 'authorization.json'
    auth.write_text(json.dumps({'scope':'isolated_pilot_only', 'policy_sha256':sha(policy),
        'protected_evidence':{str(protected):sha(protected)},
        'fixed_manifests':{'humanml3d':{'path':str(manifest),'sha256':sha(manifest)}}}))
    monkeypatch.setattr(pilot_scope, 'AUTH', auth)
    monkeypatch.setattr(pilot_scope, 'PATH', policy)
    monkeypatch.setattr(pilot_scope, 'require_frozen', lambda: None)
    pilot_scope.require_pilot('humanml3d')
    protected.write_text('{"status":"PASS"}')
    with pytest.raises(ValueError, match='PILOT_PROTECTED_EVIDENCE_CHANGED'):
        pilot_scope.require_pilot('humanml3d')
