import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from g1_pipeline.resample import time_grid,rotations,linear


@pytest.mark.parametrize('fps',[60.,90.,120.,119.9976,25.])
def test_rotation_and_translation_on_actual_25hz_grid(fps):
    t=np.arange(int(fps*1.3)+1)/fps
    out,mapping=time_grid(t,start=.017,end=1.213)
    assert out[-1]<1.213 and out[-1]<=t[-1]
    np.testing.assert_allclose(np.diff(out),.04,atol=1e-14)
    # Cross +/-pi with a moving root; Euler interpolation would take the wrong arc.
    angles=np.deg2rad(170)+t*.8
    source=Rotation.from_euler('z',angles[:,None]).as_matrix()
    target=rotations(source,mapping)
    expected=Rotation.from_euler('z',(np.deg2rad(170)+out*.8)[:,None]).as_matrix()
    np.testing.assert_allclose(target,expected,atol=1e-12)
    translation=np.stack([t,2*t,-3*t],axis=1)
    np.testing.assert_allclose(linear(translation,mapping),np.stack([out,2*out,-3*out],axis=1),atol=1e-12)
