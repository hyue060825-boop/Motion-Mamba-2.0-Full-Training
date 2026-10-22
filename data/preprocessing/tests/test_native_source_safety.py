import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from g1_pipeline.native_rest_mapping import mirror_frames
from g1_pipeline.source_action_semantics import classify


def test_mirror_exchanges_sides_reflects_world_and_preserves_so3():
    frame={'pelvis':(np.array([2.,3.,1.]),R.from_euler('z',.3).as_quat(scalar_first=True)),
           'left_wrist':(np.array([3.,4.,2.]),R.from_euler('xyz',[.2,.4,-.7]).as_quat(scalar_first=True)),
           'right_wrist':(np.array([1.,2.,.8]),R.from_euler('xyz',[-.2,.1,.5]).as_quat(scalar_first=True))}
    mirrored=mirror_frames([frame])[0]
    np.testing.assert_allclose(mirrored['left_wrist'][0],[-1.,2.,.8])
    restored=mirror_frames([mirrored])[0]
    for name in frame:
        np.testing.assert_allclose(restored[name][0],frame[name][0])
        np.testing.assert_allclose(R.from_quat(restored[name][1],scalar_first=True).as_matrix(),R.from_quat(frame[name][1],scalar_first=True).as_matrix(),atol=1e-12)


def test_scene_evidence_does_not_treat_stairs_or_unknown_seats_as_flat_ground():
    assert classify({'captions':['Walks slowly upstairs on the stairs.']})['scene_class']=='external_support_or_nonflat'
    assert classify({'captions':['Walks then sits down on something.']})['scene_class']=='support_ambiguous_review'
    assert classify({'captions':['Sits down cross-legged on the ground.']})['scene_class']=='ground_multisupport'
    assert classify({'captions':['Squats and stands up.']})['scene_class']=='flat_ground_core_candidate'


def test_smoke_checks_upstream_before_opening_or_processing_hml(monkeypatch):
    from g1_pipeline import production_hml
    def blocked(stage):
        assert stage=='01_bones_calibration'
        raise RuntimeError('EXPECTED_UPSTREAM_BLOCK')
    monkeypatch.setattr(production_hml,'require_pass',blocked)
    with pytest.raises(RuntimeError,match='EXPECTED_UPSTREAM_BLOCK'):production_hml.run()


def test_source_processor_requires_correct_source_predecessor(monkeypatch):
    from g1_pipeline import source_motion_processing
    calls=[]
    def blocked(stage):
        calls.append(stage)
        raise RuntimeError('EXPECTED_UPSTREAM_BLOCK')
    monkeypatch.setattr(source_motion_processing,'require_pass',blocked)
    for source in ['humanml3d','hiphi']:
        with pytest.raises(RuntimeError,match='EXPECTED_UPSTREAM_BLOCK'):
            source_motion_processing.process([],[],None,{},dataset=source)
    assert calls==['01_bones_calibration','02_hml100']


def test_hiphi_and_dynamic_do_not_open_inputs_before_upstream_pass(monkeypatch):
    from g1_pipeline import production_hiphi,dynamic_tracking
    def hiphi_blocked(stage):
        assert stage=='02_hml100'
        raise RuntimeError('EXPECTED_UPSTREAM_BLOCK')
    def dynamic_blocked(stage):
        assert stage=='03_hiphi100'
        raise RuntimeError('EXPECTED_UPSTREAM_BLOCK')
    monkeypatch.setattr(production_hiphi,'require_pass',hiphi_blocked)
    monkeypatch.setattr(dynamic_tracking,'require_pass',dynamic_blocked)
    with pytest.raises(RuntimeError,match='EXPECTED_UPSTREAM_BLOCK'):production_hiphi.run()
    with pytest.raises(RuntimeError,match='EXPECTED_UPSTREAM_BLOCK'):dynamic_tracking.Tracker(None)
