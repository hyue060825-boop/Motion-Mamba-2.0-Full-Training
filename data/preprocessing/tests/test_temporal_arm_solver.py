import copy
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT
from g1_pipeline.locked_target import create_retargeter
from g1_pipeline.gmr_temporal_arms import solve


def test_sequence_resets_state_and_keeps_static_target_static():
    cfgpath=ROOT/'configs/bones_temporal_arms_v15.json';cfg=json.loads(cfgpath.read_text())
    rest=json.loads((ROOT/'audit/soma_rest_mapping_v3.json').read_text())
    human={r['source']:[np.asarray(r['source_rest_position_m']),R.from_matrix(r['source_rest_rotation']).as_quat(scalar_first=True)] for r in rest['mapping']}
    frames=[copy.deepcopy(human) for _ in range(3)];outputs=[]
    for dirty in (False,True):
        solver=create_retargeter(cfgpath,ROOT/'assets/project_g1_v1/g1.xml')
        if dirty:
            q=solver.model.qpos0.copy();q[:3]=[100,-200,50];q[7:]=solver.model.jnt_range[1:,1]*.9;solver.configuration.update(q)
        q,report=solve(solver,frames,cfg['temporal_arms'],cfg['weighted_convergence']);outputs.append(q)
        assert np.isfinite(q).all()
        assert np.max(abs(np.diff(q[:,7:],axis=0)))<1e-3
        assert np.max(solver.model.jnt_range[1:,0]-q[:,7:])<=1e-4
        assert np.max(q[:,7:]-solver.model.jnt_range[1:,1])<=1e-4
        assert report['reference_input'] is False
    np.testing.assert_allclose(outputs[0],outputs[1],atol=1e-10,rtol=0)
    for frame in frames:
        for name in human:
            for actual,expected in zip(frame[name],human[name]):np.testing.assert_array_equal(actual,expected)


def test_temporal_runner_checks_preflight_before_touching_outputs(monkeypatch):
    from g1_pipeline import gates,bones_temporal_development
    def reject(stage):
        assert stage=='00_preflight'
        raise RuntimeError('EXPECTED_STALE_PREFLIGHT')
    monkeypatch.setattr(gates,'require_pass',reject)
    with pytest.raises(RuntimeError,match='EXPECTED_STALE_PREFLIGHT'):bones_temporal_development.run()
