import json
from g1_pipeline import gates


def isolated_root(tmp_path,monkeypatch):
    monkeypatch.setattr(gates,'ROOT',tmp_path)
    (tmp_path/'g1_pipeline').mkdir();(tmp_path/'configs').mkdir()
    (tmp_path/'run_pipeline.py').write_text('# test runner\n')
    (tmp_path/'g1_pipeline/example.py').write_text('version=1\n')
    (tmp_path/'configs/run.json').write_text('{}')
    return tmp_path


def test_code_config_and_inputs_change_fingerprints(tmp_path,monkeypatch):
    root=isolated_root(tmp_path,monkeypatch);source=root/'manifest.jsonl';source.write_text('{}\n')
    a=gates.gate('example','BLOCKED',input_paths=[source])
    (root/'g1_pipeline/example.py').write_text('version=2\n')
    (root/'configs/run.json').write_text('{"fps":25}')
    source.write_text('{"sample":1}\n')
    b=gates.gate('example','BLOCKED',input_paths=[source])
    for k in ('code_hash','config_hash','input_manifest_hash'): assert a[k]!=b[k]


def test_preflight_cannot_pass_before_numeric_verification(tmp_path,monkeypatch):
    root=isolated_root(tmp_path,monkeypatch)
    from g1_pipeline import audit
    monkeypatch.setattr(audit,'environment',lambda:None)
    monkeypatch.setattr(audit,'contract_definitions',lambda:None)
    def numeric_failure():
        assert json.loads((root/'gates/00_preflight.json').read_text())['status']=='RUNNING'
        return {'status':'FAIL','count':1,'scope':'intentional failed numerical check'}
    monkeypatch.setattr(gates,'numerical_contracts',numeric_failure)
    assert gates.audit()['status']=='FAIL'
    assert json.loads((root/'gates/00_preflight.json').read_text())['status']=='FAIL'


def test_pass_with_changed_input_cannot_start_next_stage(tmp_path,monkeypatch):
    import pytest
    root=isolated_root(tmp_path,monkeypatch)
    source=root/'panel.jsonl';source.write_text('{"id":1}\n')
    gates.gate('00_preflight','PASS',input_paths=[source])
    assert gates.require_pass('00_preflight')['status']=='PASS'
    source.write_text('{"id":2}\n')
    with pytest.raises(RuntimeError,match='GATE_INPUT_STALE'):
        gates.require_pass('00_preflight')


def test_independent_additions_do_not_invalidate_but_wiring_them_does(tmp_path,monkeypatch):
    import pytest
    root=isolated_root(tmp_path,monkeypatch)
    gates.gate('00_preflight','PASS')
    (root/'g1_pipeline/later_stage.py').write_text('version=1\n')
    (root/'configs/later_stage.json').write_text('{}')
    assert gates.require_pass('00_preflight')['status']=='PASS'
    (root/'g1_pipeline/example.py').write_text('from . import later_stage\n')
    with pytest.raises(RuntimeError,match='GATE_FINGERPRINT_STALE'):
        gates.require_pass('00_preflight')


def test_removing_bound_dependency_invalidates_gate(tmp_path,monkeypatch):
    import pytest
    root=isolated_root(tmp_path,monkeypatch)
    gates.gate('00_preflight','PASS')
    (root/'configs/run.json').unlink()
    with pytest.raises(RuntimeError,match='GATE_DEPENDENCY_MISSING'):
        gates.require_pass('00_preflight')
