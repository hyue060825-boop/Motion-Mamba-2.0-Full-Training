import json
import numpy as np
import pytest
from g1_pipeline import release_validation as release
from g1_pipeline.common import write_json,sha


def test_upstream_pass_and_placeholder_files_cannot_release(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'ROOT',tmp_path)
    monkeypatch.setattr(release,'require_pass',lambda stage:{'status':'PASS'})
    monkeypatch.setattr(release,'fingerprint',lambda:{'code_hash':'test'})
    recorded=[];monkeypatch.setattr(release,'gate',lambda *args,**kw:recorded.append((args,kw)))
    for name in ('manifests/accepted_train.jsonl','stats/physical_414.json','reports/leakage_audit.md'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\n')
    report=release.run()
    assert report['status']=='BLOCKED' and not report['release_allowed']
    assert recorded[0][0][1]=='BLOCKED'
    assert (tmp_path/'reports/final_acceptance.md').exists()


def test_wrong_model_or_stale_runtime_evidence_rejected_before_loading_arrays(tmp_path):
    path=tmp_path/'qa/aggregate/final_runtime_acceptance.json'
    evidence={'schema_version':1,'fingerprint':{'code_hash':'old'},'model_contract':release.MODEL_CONTRACT}
    write_json(path,evidence)
    with pytest.raises(ValueError,match='STALE'):release.runtime_evidence(tmp_path,{'code_hash':'new'})
    evidence['fingerprint']={'code_hash':'new'};evidence['model_contract']={**release.MODEL_CONTRACT,'representation':330}
    write_json(path,evidence)
    with pytest.raises(ValueError,match='G1_MODEL_CONTRACT'):release.runtime_evidence(tmp_path,{'code_hash':'new'})


def test_bones_calibration_candidate_cannot_enter_main_training_pool(tmp_path):
    row={'sample_id':'bones:x','dataset':'bones','source_family_id':'bones:take',
         'project_split':'train','status':'QA_PASS','target_fps':25,'sim_qa_status':'PASS',
         'leakage_status':'PASS','gmr_role':'calibration_candidate'}
    p=tmp_path/'manifests/accepted_train.jsonl';p.parent.mkdir();p.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='BONES_MAIN_POOL_NOT_OFFICIAL'):release.manifest_evidence(tmp_path)
