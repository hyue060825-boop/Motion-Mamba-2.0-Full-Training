import numpy as np
from g1_pipeline.ground_clearance import smooth_lift


def test_valid_ground_is_identity_and_constant_required_lift_is_exact():
    np.testing.assert_array_equal(smooth_lift(np.full(10,-.01)),np.zeros(10))
    np.testing.assert_allclose(smooth_lift(np.full(10,.03)),np.full(10,.03),atol=1e-6)


def test_short_penetration_is_bounded_and_smoothed_without_shortening():
    lower=np.zeros(21);lower[10]=.05
    result=smooth_lift(lower)
    assert result.shape==lower.shape and np.all(result>=lower)
    assert result[9]>0 and result[11]>0
    assert max(abs(np.diff(result)))<max(abs(np.diff(lower)))
