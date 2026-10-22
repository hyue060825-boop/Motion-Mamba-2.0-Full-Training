"""Offline closest hinge trajectory subject to explicit rate and angle bounds.

This changes poses, never timestamps. Chosen rate bounds are experimental
simulation-reference limits, not verified hardware capabilities.
"""
import numpy as np
from scipy import sparse
from qpsolvers import solve_qp


def project(values, lower, upper, max_speed, *, dt=.04, correction_curvature=1.):
    values=np.asarray(values,dtype=float);n,j=values.shape
    if n<2 or not np.isfinite(values).all() or dt<=0:
        raise ValueError('INVALID_TRAJECTORY')
    lower=np.broadcast_to(lower,(j,));upper=np.broadcast_to(upper,(j,));max_speed=np.broadcast_to(max_speed,(j,))
    if not all(np.isfinite(x).all() for x in (lower,upper,max_speed)) or np.any(lower>=upper) or np.any(max_speed<=0) or correction_curvature<0:
        raise ValueError('INVALID_TRAJECTORY_BOUNDS')
    d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc')
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    p=sparse.eye(n,format='csc')+correction_curvature*(d2.T@d2)
    g=sparse.vstack([d1,-d1],format='csc');result=values.copy();changed=[]
    for k in range(j):
        v=values[:,k]
        if np.all((v>=lower[k])&(v<=upper[k])) and np.max(abs(np.diff(v)))<=max_speed[k]*dt+1e-10:
            continue
        x=solve_qp(p,-p@v,g,np.full(2*(n-1),max_speed[k]*dt),
                   lb=np.full(n,lower[k]),ub=np.full(n,upper[k]),solver='proxqp',
                   eps_abs=1e-9,eps_rel=1e-9,max_iter=10000,
                   check_duality_gap=True,eps_duality_gap_abs=1e-10,eps_duality_gap_rel=1e-10)
        if x is None or not np.isfinite(x).all():raise ValueError('RATE_PROJECTION_QP_FAILED')
        if np.max(abs(np.diff(x)))>max_speed[k]*dt+1e-7 or np.min(x)<lower[k]-1e-7 or np.max(x)>upper[k]+1e-7:
            raise ValueError('RATE_PROJECTION_CONSTRAINT_FAILED')
        result[:,k]=x;changed.append(k)
    return result,{'changed_joint_indices':changed,'correction_mae_rad':float(abs(result-values).mean()),
                   'correction_max_rad':float(abs(result-values).max()),'dt_s':dt,
                   'correction_curvature':correction_curvature,'max_speed_rad_s':max_speed.tolist(),
                   'objective':'squared pose correction + correction_curvature * squared second difference of pose correction',
                   'time_changed':False,'hardware_acceptance':False}


def sonic_training_reference_limits(model):
    """Transcription of SONIC training G1 config; checked by the audit runner."""
    limits=[]
    for j in range(1,model.njnt):
        name=model.joint(j).name
        if any(s in name for s in ('hip_pitch','hip_roll','knee')):value=20.
        elif any(s in name for s in ('hip_yaw','waist_yaw')):value=32.
        elif any(s in name for s in ('wrist_pitch','wrist_yaw')):value=22.
        elif any(s in name for s in ('ankle','waist_roll','waist_pitch','shoulder','elbow','wrist_roll')):value=37.
        else:raise ValueError('UNMAPPED_REFERENCE_RATE:'+name)
        limits.append(value)
    return np.array(limits)
