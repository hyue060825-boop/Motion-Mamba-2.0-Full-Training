"""Minimal smooth upward root correction against the locked floor geometry.

No reference trajectory, time warp, or root-height canonicalization is used.
The complete original trajectory and per-frame lift must be retained by callers.
"""
import numpy as np
import mujoco
from scipy import sparse
from qpsolvers import solve_qp


def smooth_lift(required, *, dt=.04, velocity_weight=.08, acceleration_weight=.004):
    required=np.asarray(required,dtype=float)
    if required.ndim!=1 or len(required)<3 or not np.isfinite(required).all() or dt<=0:
        raise ValueError('INVALID_CLEARANCE_REQUIREMENT')
    lower=np.maximum(required,0.);n=len(lower)
    if not lower.any():return lower
    d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc')/dt
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')/dt**2
    h=sparse.eye(n,format='csc')+velocity_weight**2*(d1.T@d1)+acceleration_weight**2*(d2.T@d2)
    result=solve_qp(h,np.zeros(n),lb=lower,solver='proxqp',eps_abs=1e-9,eps_rel=1e-9,
                    check_duality_gap=True,eps_duality_gap_abs=1e-10,eps_duality_gap_rel=1e-10,max_iter=10000)
    if result is None or not np.isfinite(result).all() or np.min(result-lower)<-1e-7:
        raise ValueError('CLEARANCE_QP_FAILED')
    return np.maximum(result,lower)


def correct(model,q, *, permitted_penetration=.015):
    if q.ndim!=2 or q.shape[1]!=36 or not np.isfinite(q).all():raise ValueError('INVALID_CLEARANCE_TRAJECTORY')
    data=mujoco.MjData(model);floor=model.geom('floor').id;depth=[]
    if model.geom_type[floor]!=mujoco.mjtGeom.mjGEOM_PLANE:raise ValueError('NONPLANAR_FLOOR')
    # Locked floor is horizontal at z=0. Reject other geometry instead of
    # assuming that a vertical translation changes depth by the same amount.
    data.qpos[:]=q[0];mujoco.mj_forward(model,data)
    if not np.allclose(data.geom_xmat[floor].reshape(3,3)[:,2],[0,0,1],atol=1e-10):raise ValueError('NONHORIZONTAL_FLOOR')
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        depth.append(max([-float(c.dist) for c in data.contact if floor in (c.geom1,c.geom2)]+[0.]))
    depth=np.array(depth);lift=smooth_lift(depth-permitted_penetration);result=q.copy();result[:,2]+=lift
    return result,{'lift_m':lift.tolist(),'original_penetration_m':depth.tolist(),
                   'lift_max_m':float(lift.max()),'lift_mean_m':float(lift.mean()),
                   'added_vertical_speed_max_m_s':float(np.max(abs(np.diff(lift)/.04))),
                   'added_vertical_acceleration_max_m_s2':float(np.max(abs(np.diff(lift,n=2)/.04**2))),
                   'permitted_penetration_m':permitted_penetration,'velocity_weight':.08,'acceleration_weight':.004,
                   'reference_input':False,'time_changed':False,'height_error_must_remain_in_paired_metrics':True}
