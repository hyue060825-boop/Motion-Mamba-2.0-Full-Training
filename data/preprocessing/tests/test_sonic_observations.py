import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from g1_pipeline.sonic_observations import SonicObservations


def test_g1_encoder_time_window_joint_order_and_relative_orientation():
    obs=SonicObservations();n=60
    q=np.arange(n)[:,None]*100+np.arange(29)[None,:]
    v=-q;quat=np.tile(R.from_euler('z',90,degrees=True).as_quat(scalar_first=True),(n,1))
    value=obs.encoder(q,v,quat,1,np.array([1.,0,0,0]),timestamps=np.arange(n)*.02)[0]
    assert value.shape==(1762,)
    target=value[obs.encoder_slices['motion_joint_positions_10frame_step5']].reshape(10,29)
    # IsaacLab index 1 is right hip pitch, MuJoCo index 6; index 2 is waist yaw.
    np.testing.assert_allclose(target[:,1],q[1+np.arange(10)*5,6])
    np.testing.assert_allclose(target[:,2],q[1+np.arange(10)*5,12])
    rotation=value[obs.encoder_slices['motion_anchor_orientation_10frame_step5']].reshape(10,6)
    np.testing.assert_allclose(rotation,np.tile([0,-1,1,0,0,0],(10,1)),atol=1e-7)
    allowed=np.zeros(1762,dtype=bool)
    for name in ('motion_joint_positions_10frame_step5','motion_joint_velocities_10frame_step5','motion_anchor_orientation_10frame_step5'):allowed[obs.encoder_slices[name]]=True
    assert not np.any(value[~allowed])
    final=obs.encoder(q,v,quat,n-1,quat[0],timestamps=np.arange(n)*.02)[0][obs.encoder_slices['motion_joint_positions_10frame_step5']].reshape(10,29)
    np.testing.assert_array_equal(final,np.tile(q[-1,obs.to_policy],(10,1)))


def test_history_is_oldest_first_gravity_is_body_local_and_action_mapping_inverts():
    obs=SonicObservations();default=obs.constants['default_angles'];tilt=R.from_euler('x',90,degrees=True).as_quat(scalar_first=True)
    for tick in range(12):obs.push_state(default+tick,np.full(29,tick*2),tilt,np.array([tick,0,0]),np.full(29,-tick),time_s=tick*.02)
    value=obs.decoder(np.zeros(64))[0]
    history=value[obs.decoder_slices['his_body_joint_positions_10frame_step1']].reshape(10,29)
    np.testing.assert_allclose(history,np.tile(np.arange(2,12)[:,None],(1,29)))
    gravity=value[obs.decoder_slices['his_gravity_dir_10frame_step1']].reshape(10,3)
    np.testing.assert_allclose(gravity,np.tile([0,-1,0],(10,1)),atol=1e-7)
    action=np.arange(29,dtype=float);target=obs.joint_target(action)
    recovered=((target-default)/obs.constants['g1_action_scale'])[obs.to_policy]
    np.testing.assert_allclose(recovered,action,atol=1e-12)


def test_decoder_refuses_missing_state_history():
    with pytest.raises(ValueError,match='HISTORY_NOT_INITIALIZED'):SonicObservations().decoder(np.zeros(64))


def test_25hz_motion_cannot_be_mislabeled_as_controller_50hz():
    obs=SonicObservations();quat=np.tile([1.,0,0,0],(3,1))
    with pytest.raises(ValueError,match='PHYSICAL_50HZ'):
        obs.encoder(np.zeros((3,29)),np.zeros((3,29)),quat,0,quat[0],timestamps=np.arange(3)*.04)
