import numpy as np
from g1_pipeline.bounded_hinge_trajectory import project


def test_two_frames_have_known_symmetric_closest_solution():
    result,_=project(np.array([[0.],[1.]]),[-2.],[2.],[.2],dt=1.)
    np.testing.assert_allclose(result[:,0],[.4,.6],atol=1e-7)


def test_legal_trajectory_is_identity_and_projection_is_idempotent():
    x=np.array([[0.,.1],[1.,.2],[0.,.3],[1.,.4]])
    result,report=project(x,[-1.,-1.],[1.,1.],[.2,1.],dt=1.)
    np.testing.assert_array_equal(result[:,1],x[:,1])
    assert max(abs(np.diff(result[:,0])))<=.2+1e-7
    again,_=project(result,[-1.,-1.],[1.,1.],[.2,1.],dt=1.)
    np.testing.assert_allclose(again,result,atol=1e-7)
    assert report['time_changed'] is False
