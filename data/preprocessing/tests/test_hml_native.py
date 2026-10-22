import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from g1_pipeline.common import HML
from g1_pipeline.hml_native import resample_local_fk,native_pose_to_frames,SMPLH_IDS
from g1_pipeline.hml_native_runner import native_parameters,estimate_rest_height
from g1_pipeline.lineage import hml_native_interval


def test_native_fk_slerp_preserves_offset_and_physical_time():
    st=np.arange(6)*.02;target=np.arange(3)*.04
    local=np.tile(np.eye(3),(len(st),2,1,1));local[:,0]=Rotation.from_rotvec(np.c_[np.zeros((len(st),2)),st*np.pi/2/.1]).as_matrix()
    trans=np.c_[st,np.zeros((len(st),2))];parents=np.array([-1,0]);off=np.array([[.1,.2,.3],[0,1,0.]])
    c=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    frames=resample_local_fk(local,trans,parents,st,target,offsets=off,names={'Hips':0,'Hand':1},world_basis=c)
    root=np.array([.14,.2,.3]);child=root+[-np.sin(np.deg2rad(36)),np.cos(np.deg2rad(36)),0]
    np.testing.assert_allclose(frames[1]['Hips'][0],c@root,atol=1e-12)
    np.testing.assert_allclose(frames[1]['Hand'][0],c@child,atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(frames[1]['Hips'][1],scalar_first=True).as_matrix(),c@Rotation.from_rotvec([0,0,np.deg2rad(36)]).as_matrix(),atol=1e-12)


def test_native_fk_rejects_invalid_times_and_root_parent():
    local=np.tile(np.eye(3),(3,1,1,1));trans=np.zeros((3,3));off=np.zeros((1,3));st=np.arange(3)*.04
    for target,error in [([-.01,.03],'TARGET_TIME_OUTSIDE'),([0,.05],'TARGET_NOT_25HZ'),([], 'INVALID_NATIVE_TIMES')]:
        with pytest.raises(ValueError,match=error):resample_local_fk(local,trans,[-1],st,target,offsets=off,names={'pelvis':0})
    with pytest.raises(ValueError,match='INVALID_NATIVE_PARENTS'):
        resample_local_fk(local,trans,[0],st,[0,.04],offsets=off,names={'pelvis':0})
    single=resample_local_fk(local[:1],trans[:1],[-1],[0],[0],offsets=off,names={'pelvis':0})
    np.testing.assert_array_equal(single[0]['pelvis'][0],[0,0,0])


def test_hml_interval_clips_processed_partial_stride_to_real_native_support():
    interval=hml_native_interval('ACCAD/Female',0,86,120.,511)
    assert interval['requested_native_end_index_exclusive']==516
    assert interval['native_end_index_exclusive']==511
    assert interval['native_end_clipped'] is True
    assert interval['last_sample_sec']==510/120.


@pytest.mark.parametrize('gender,fps',[('female',120.),('male',90.)])
def test_native_shape_dmpl_and_all_joints_match_independent_bodymodel(tmp_path,monkeypatch,gender,fps):
    monkeypatch.syspath_prepend(str(HML))
    from human_body_prior.body_model.body_model import BodyModel
    rng=np.random.default_rng(71);st=np.arange(20)/fps;target=np.array([.04,.08,.12])
    # Analytic constant local angular velocities, linear translation and DMPLs.
    # Expected poses at target timestamps do not call adapter interpolation.
    omega=rng.normal(size=(52,3))*.8;omega[0]=[.8,-.3,2.]
    trans0=np.array([1.,-.5,.2]);velocity=np.array([.3,-.1,.05]);beta=rng.normal(size=16)*.1
    dmpl0=rng.normal(size=8)*.05;dmpl_v=rng.normal(size=8)*.3
    path=tmp_path/'sample.npz'
    np.savez(path,poses=(st[:,None,None]*omega).reshape(len(st),156),trans=trans0+st[:,None]*velocity,
             dmpls=dmpl0+st[:,None]*dmpl_v,betas=beta,gender=gender,mocap_framerate=fps)
    p=native_parameters(path,start_frame=1,end_frame=len(st))
    names={str(i):i for i in range(52)}
    frames=native_pose_to_frames(p['full_pose'],p['trans'],p['parents'],p['offsets'],fps,target,start_frame=1,names=names)
    import torch
    bm=BodyModel(bm_fname=str(HML/f'body_models/smplh/{gender}/model.npz'),dmpl_fname=str(HML/f'body_models/dmpls/{gender}/model.npz'),num_betas=16,num_dmpls=8)
    poses=(target[:,None,None]*omega).reshape(len(target),156)
    with torch.no_grad():
        out=bm(root_orient=torch.tensor(poses[:,:3]).float(),pose_body=torch.tensor(poses[:,3:66]).float(),pose_hand=torch.tensor(poses[:,66:]).float(),
               betas=torch.tensor(np.tile(beta,(len(target),1))).float(),dmpls=torch.tensor(dmpl0+target[:,None]*dmpl_v).float(),trans=torch.tensor(trans0+target[:,None]*velocity).float())
    actual=np.array([[f[str(i)][0] for i in range(52)] for f in frames])
    np.testing.assert_allclose(actual,out.Jtr.numpy(),atol=2e-6,rtol=0)
    # Arm anatomy is fixed by the model topology, not names renamed to SOMA.
    for side in ('left','right'):
        shoulder,elbow,wrist=[SMPLH_IDS[f'{side}_{x}'] for x in ('shoulder','elbow','wrist')]
        assert p['parents'][wrist]==elbow and p['parents'][elbow]==shoulder
    # Isolated 29-joint robot mapping has no role in this native geometry test.
    zero=torch.zeros((1,3))
    with torch.no_grad():rest=bm(root_orient=zero,pose_body=torch.zeros((1,63)),pose_hand=torch.zeros((1,90)),betas=torch.tensor(beta[None]).float(),dmpls=torch.zeros((1,8)),trans=zero)
    np.testing.assert_allclose(estimate_rest_height(path),np.ptp(rest.v.numpy()[0,:,1]),atol=2e-6)
    assert abs(estimate_rest_height(path)-np.ptp(rest.Jtr.numpy()[0,:,1]))>.05
