import json
from g1_pipeline.common import ROOT


def test_v6_body_residual_evidence_has_all_targets_and_full_denominator():
    report=json.loads((ROOT/'qa/aggregate/bones_soma_rootweight_dev_v6_body_residuals.json').read_text())
    assert report['clips']==200
    assert len(report['body_residuals'])==14
    assert all(v['clips']==200 for v in report['body_residuals'].values())
