"""Bounded arm warm-start search for GMR, using human targets only.

Development experiment. Root/torso/legs remain at the current candidate during
seed search. A reference robot pose is never an input to this module.
"""
import numpy as np
from scipy.optimize import least_squares


def arm_seed(solver,side,max_nfev=100):
    if side not in ('left','right'):raise ValueError('INVALID_ARM_SIDE')
    model=solver.model;configuration=solver.configuration;base=configuration.q.copy()
    suffixes=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
    ids=[model.joint(f'{side}_{name}_joint').id for name in suffixes];adr=model.jnt_qposadr[ids]
    lower=model.jnt_range[ids,0]+1e-7;upper=model.jnt_range[ids,1]-1e-7
    names=[f'{side}_{part}_link' for part in ('shoulder_yaw','elbow','wrist_yaw')]
    tasks=[solver.human_body_to_task2[solver.ik_match_table2[name][0]] for name in names]
    def residual(x):
        pose=base.copy();pose[adr]=x;configuration.update(pose)
        return np.concatenate([task.cost*task.compute_error(configuration) for task in tasks])
    initial_error=residual(np.clip(base[adr],lower,upper));initial=float(initial_error@initial_error)
    starts=[np.clip(base[adr],lower,upper),np.clip(model.qpos0[adr],lower,upper)]
    for fraction in (.2,.5,.8):
        start=starts[0].copy();start[:3]=lower[:3]+fraction*(upper[:3]-lower[:3]);starts.append(start)
    fits=[]
    try:
        for start in starts:
            fit=least_squares(residual,start,bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            fits.append(fit)
        best=min(fits,key=lambda f:float(f.fun@f.fun));pose=base.copy();best_cost=float(best.fun@best.fun)
        if best_cost<initial:pose[adr]=best.x
        configuration.update(pose)
        return pose,{'side':side,'initial_arm_cost':initial,'best_arm_cost':min(best_cost,initial),
                     'seed_count':len(starts),'best_solver_success':bool(best.success),
                     'max_hinge_change_deg':float(np.rad2deg(abs(pose[adr]-base[adr])).max()),
                     'evaluations':sum(f.nfev for f in fits),'uses_reference_pose':False}
    except Exception:
        configuration.update(base);raise
