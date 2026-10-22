"""Pose-only positions decoder for scalar losses that only consume posed_joints.

Copy the official positions expressions exactly, omitting unrelated rotations
which are not used by these callers. Rotation-derived FK remains unchanged.
This context is never used for history, generation, serving or other callers.
"""
import contextlib, types
import einops
import torch

@contextlib.contextmanager
def positions_only(rep, enabled):
    if not enabled:
        yield;return
    original=rep.inverse
    def inverse(self,features,is_normalized,posed_joints_from='rotations',return_numpy=False):
        if posed_joints_from!='positions' or return_numpy:
            return original(features,is_normalized,posed_joints_from,return_numpy)
        assert features.ndim==3
        if is_normalized:features=self.unnormalize(features)
        root_positions,heading,local_positions,rotation,velocities,contacts=einops.unpack(features,self.ps,'batch time *')
        dummy_root=torch.zeros_like(local_positions[:,:,[0]])
        posed=torch.cat([dummy_root,local_positions],dim=2)
        posed[...,0]+=root_positions[...,None,0]
        posed[...,2]+=root_positions[...,None,2]
        return dict(posed_joints=posed,root_positions=root_positions)
    rep.inverse=types.MethodType(inverse,rep)
    try:yield
    finally:rep.inverse=original
