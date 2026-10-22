"""Native SMPL-H local SO(3) resampling and FK, before any robot mapping.

World basis is explicit. AMASS root poses already define the capture world;
no BVH Y-up conversion or SOMA body-frame offsets are silently applied.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Shared SMPL-H/SMPL-X body landmarks. 22+ are SMPL-H fingers, not wrists.
SMPLH_IDS={'pelvis':0,'spine3':9,'left_hip':1,'left_knee':4,'left_ankle':7,
           'right_hip':2,'right_knee':5,'right_ankle':8,'left_shoulder':16,
           'left_elbow':18,'left_wrist':20,'right_shoulder':17,
           'right_elbow':19,'right_wrist':21}


def _check_times(times, target):
    times=np.asarray(times,float);target=np.asarray(target,float)
    if times.ndim!=1 or target.ndim!=1 or not len(times) or not len(target) or not np.isfinite(times).all() or not np.isfinite(target).all():
        raise ValueError('INVALID_NATIVE_TIMES')
    if np.any(np.diff(times)<=0) or np.any(np.diff(target)<=0):raise ValueError('NON_MONOTONIC_NATIVE_TIMES')
    if target[0]<times[0]-1e-10 or target[-1]>times[-1]+1e-10:raise ValueError('TARGET_TIME_OUTSIDE_NATIVE_SUPPORT')
    if len(target)>1 and np.max(np.abs(np.diff(target)-.04))>1e-6:raise ValueError('TARGET_NOT_25HZ')
    return times,np.clip(target,times[0],times[-1])


def sample_local(local_rotations, root_translation, offsets, source_times, target_times):
    """Sample local rotations, world translation, and shape/DMPL rest offsets."""
    st,tt=_check_times(source_times,target_times)
    local=np.asarray(local_rotations,float);trans=np.asarray(root_translation,float)
    if local.ndim!=4 or local.shape[0]!=len(st) or local.shape[2:]!=(3,3) or trans.shape!=(len(st),3):raise ValueError('INVALID_NATIVE_POSE_SHAPES')
    if not np.isfinite(local).all() or not np.isfinite(trans).all():raise ValueError('NONFINITE_NATIVE_POSE')
    if not np.allclose(local@local.swapaxes(-1,-2),np.eye(3),atol=1e-6) or np.any(np.abs(np.linalg.det(local)-1)>1e-6):raise ValueError('NON_SO3_NATIVE_ROTATIONS')
    if offsets is None:raise ValueError('NATIVE_OFFSETS_REQUIRED')
    offsets=np.asarray(offsets,float);j=local.shape[1]
    if offsets.shape==(j,3):offsets=np.broadcast_to(offsets,(len(st),j,3))
    if offsets.shape!=(len(st),j,3) or not np.isfinite(offsets).all():raise ValueError('INVALID_NATIVE_OFFSETS')
    right=np.searchsorted(st,tt).clip(0,len(st)-1);left=np.maximum(0,right-1)
    width=st[right]-st[left];w=np.divide(tt-st[left],width,out=np.zeros(len(tt)),where=width>0)
    def linear(a):
        v=w.reshape((-1,)+(1,)*(a.ndim-1));return a[left]*(1-v)+a[right]*v
    if len(st)==1:interp=np.repeat(local,len(tt),axis=0)
    else:interp=np.stack([Slerp(st,Rotation.from_matrix(local[:,k]))(tt).as_matrix() for k in range(j)],axis=1)
    return interp,linear(trans),linear(offsets)


def forward_kinematics(local, trans, parents, offsets, world_basis=None):
    """Offsets[0] is the shaped pelvis location; trans is AMASS translation.

    Only the world basis changes: R_world'=C R_world. Native local body axes
    remain intact for later, separately validated robot frame corrections.
    """
    parents=np.asarray(parents);j=local.shape[1]
    if parents.shape!=(j,) or not np.issubdtype(parents.dtype,np.integer) or parents[0]!=-1 or np.any((parents[1:]<0)|(parents[1:]>=np.arange(1,j))):raise ValueError('INVALID_NATIVE_PARENTS')
    c=np.eye(3) if world_basis is None else np.asarray(world_basis,float)
    if c.shape!=(3,3) or not np.allclose(c@c.T,np.eye(3),atol=1e-10) or not np.isclose(np.linalg.det(c),1):raise ValueError('INVALID_WORLD_BASIS')
    wr=np.empty_like(local);wp=np.empty_like(offsets)
    wr[:,0]=local[:,0];wp[:,0]=trans+offsets[:,0]
    for k in range(1,j):
        p=parents[k];wr[:,k]=wr[:,p]@local[:,k]
        wp[:,k]=wp[:,p]+np.einsum('nij,nj->ni',wr[:,p],offsets[:,k])
    return wp@c.T,np.einsum('ij,nkjl->nkil',c,wr)


def resample_local_fk(local_rotations, root_translation, parents, source_times, target_times,
                      *, offsets=None, names=SMPLH_IDS, world_basis=None):
    local,trans,off=sample_local(local_rotations,root_translation,offsets,source_times,target_times)
    if any(not isinstance(k,(int,np.integer)) or not 0<=k<local.shape[1] for k in names.values()):raise ValueError('SEMANTIC_JOINT_OUT_OF_RANGE')
    wp,wr=forward_kinematics(local,trans,parents,off,world_basis)
    return [{name:(wp[i,k],Rotation.from_matrix(wr[i,k]).as_quat(scalar_first=True)) for name,k in names.items()} for i in range(len(wp))]


def native_pose_to_frames(full_pose, trans, parents, offsets, source_fps, target_times,
                          start_frame=0, names=SMPLH_IDS, world_basis=None):
    full=np.asarray(full_pose,float);trans=np.asarray(trans,float)
    if full.ndim!=3 or full.shape[2]!=3 or trans.shape!=(len(full),3):raise ValueError('INVALID_SMPLH_POSE')
    if not np.isfinite(source_fps) or source_fps<=0 or start_frame<0:raise ValueError('INVALID_NATIVE_FPS_OR_START')
    times=(start_frame+np.arange(len(full)))/float(source_fps)
    local=Rotation.from_rotvec(full.reshape(-1,3)).as_matrix().reshape(full.shape[:2]+(3,3))
    return resample_local_fk(local,trans,parents,times,target_times,offsets=offsets,names=names,world_basis=world_basis)
