"""Joint root-height/hinge optimization against human tasks and locked geometry.

Only root z and bounded hinges move. Root xy/orientation and all timestamps
remain fixed. The optimizer never reads the official robot reference.
"""
import numpy as np
from scipy.optimize import least_squares
from .gmr_collision_refinement import collision_pairs,collision_terms


def floor_pairs(model):
    floor=model.geom('floor').id;pairs=set()
    for g in range(model.ngeom):
        if model.geom_bodyid[g]==0:continue
        if ((model.geom_contype[floor]&model.geom_conaffinity[g]) or
                (model.geom_contype[g]&model.geom_conaffinity[floor])):
            pairs.add((floor,g))
    for a,b in zip(model.pair_geom1,model.pair_geom2):
        if floor in (a,b):pairs.add((floor,int(b if a==floor else a)))
    return sorted(pairs)


def refine(solver,frame,seed,previous_correction=None,max_nfev=100):
    solver.update_targets(frame,offset_to_ground=False);m=solver.model;c=solver.configuration
    if (m.nq,m.nv)!=(36,35):raise ValueError('UNEXPECTED_TARGET_CONTRACT')
    qadr=np.r_[2,np.arange(7,36)];vadr=np.r_[2,np.arange(6,35)]
    anchor=seed[qadr].copy();q=seed.copy();self_pairs=collision_pairs(m);ground_pairs=floor_pairs(m)
    if previous_correction is not None and (np.shape(previous_correction)!=(30,) or not np.isfinite(previous_correction).all()):
        raise ValueError('INVALID_GROUND_TEMPORAL_STATE')
    posture=np.r_[75.,np.full(29,5.)];temporal=np.r_[1.5,np.full(29,.4)]/.04
    lower=np.r_[seed[2]-.15,m.jnt_range[1:,0]];upper=np.r_[seed[2]+.15,m.jnt_range[1:,1]]
    cached=None;values=None
    def evaluate(x):
        nonlocal cached,values
        if cached is not None and np.array_equal(x,cached):return values
        q[qadr]=x;c.update(q);residuals=[];jacobians=[]
        for task in solver.tasks2:
            residuals.append(task.cost*task.compute_error(c))
            jacobians.append(task.cost[:,None]*task.compute_jacobian(c)[:,vadr])
        for pairs,minimum in [(self_pairs,-.002),(ground_pairs,-.015)]:
            error,jac=collision_terms(m,c.data,pairs,minimum=minimum)
            residuals.append(2000.*error);jacobians.append(2000.*jac[:,vadr])
        residuals.append(posture*(x-anchor));jacobians.append(np.diag(posture))
        if previous_correction is not None:
            residuals.append(temporal*(x-anchor-previous_correction));jacobians.append(np.diag(temporal))
        cached=x.copy();values=np.concatenate(residuals),np.concatenate(jacobians)
        return values
    initial=anchor if previous_correction is None else anchor+previous_correction
    initial=np.clip(initial,lower+1e-12,upper-1e-12)
    fit=least_squares(lambda x:evaluate(x)[0],initial,jac=lambda x:evaluate(x)[1],bounds=(lower,upper),
                      max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    q[qadr]=fit.x;c.update(q)
    self_error=collision_terms(m,c.data,self_pairs,minimum=-.002)[0]
    ground_error=collision_terms(m,c.data,ground_pairs,minimum=-.015)[0]
    return q.copy(),{'nfev':fit.nfev,'optimizer_success':bool(fit.success),'objective':float(2*fit.cost),
                     'ground_excess_max_m':float(ground_error.max(initial=0)),
                     'self_excess_max_m':float(self_error.max(initial=0)),
                     'root_z_change_m':float(q[2]-seed[2]),
                     'hinge_correction_max_deg':float(np.rad2deg(abs(q[7:]-seed[7:])).max()),
                     'floor_pair_count':len(ground_pairs),'reference_input':False,'time_changed':False}
