import numpy as np
from g1_pipeline.bones_relative_motion import labels, threshold


def test_metadata_multilabel_preserves_turning_locomotion():
    assert labels({'category': 'walking, turning'}) == ['locomotion', 'turning']
    assert labels({'category': 'swimming'}) == []


def test_zero_variability_has_nonzero_margin_and_coordinate_independence():
    x = np.tile([2., 0.], (20, 1))
    np.testing.assert_allclose(threshold(x, .01), [2.03, .03])
    # A moving joint must not set the threshold for another stationary joint.
    x[:, 0] = np.arange(20)
    assert threshold(x, .01)[0] > 19
    assert threshold(x, .01)[1] == .03
