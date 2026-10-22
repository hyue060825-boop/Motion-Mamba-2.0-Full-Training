from pathlib import Path

from g1_pipeline.common import ROOT


def test_hml_and_hiphi_do_not_expose_historical_unitree_runner():
    for name in ('hml_smoke.py','hiphi_smoke.py'):
        text=(ROOT/'g1_pipeline'/name).read_text()
        assert "GeneralMotionRetargeting('bvh_lafan1','unitree_g1'" not in text
        assert "GeneralMotionRetargeting('bvh_hiphi','unitree_g1'" not in text
        assert 'locked project G1' in text
        assert 'require_pass' in text
