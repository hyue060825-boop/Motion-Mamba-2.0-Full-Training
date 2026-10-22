import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from g1_pipeline.initial_planar_alignment import initial_alignment


def pose(rotation,position=(0,0,.8)):
    q=np.zeros(36);q[:3]=position;q[3:7]=rotation.as_quat(scalar_first=True);return q


@pytest.mark.parametrize('pitch',[0.,90.,-90.,179.])
def test_initial_yaw_recovered_even_when_forward_axis_is_vertical(pitch):
    candidate=pose(R.from_rotvec([0,np.deg2rad(pitch),0]),(1.,2.,.8))
    turn=R.from_rotvec([0,0,1.7]);reference=candidate.copy()
    reference[:3]=turn.apply(candidate[:3])+[3,-4,.2]
    reference[3:7]=(turn*R.from_quat(candidate[3:7],scalar_first=True)).as_quat(scalar_first=True)
    a=initial_alignment(candidate,reference)
    np.testing.assert_allclose(a['yaw_rad'],1.7,atol=1e-12)
    np.testing.assert_allclose(a['translation'],[3,-4,0],atol=1e-12)
    assert a['frames_used']==[0]
    reference[3:7]*=-1
    np.testing.assert_allclose(initial_alignment(candidate,reference)['yaw_rad'],1.7,atol=1e-12)


def test_unobservable_opposite_up_directions_are_rejected():
    with pytest.raises(ValueError,match='INITIAL_YAW_UNOBSERVABLE'):
        initial_alignment(pose(R.identity()),pose(R.from_rotvec([np.pi,0,0])))
