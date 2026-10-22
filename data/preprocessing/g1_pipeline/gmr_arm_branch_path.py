"""Enumerate target shoulder branches and choose a whole-clip continuous path."""
import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .gmr_arm_trajectory_seed import se3_error


def shoulder_branches(chain,parent_rotation,target_rotation):
    if not np.allclose(chain.axes[:3],[[0,1,0],[1,0,0],[0,0,1]],atol=1e-12):raise ValueError('UNSUPPORTED_SHOULDER_AXES')
    restvec=R.from_matrix(chain.rest[1]).as_rotvec()
    if np.max(abs(restvec[1:]))>1e-12 or not np.allclose(chain.rest[2],np.eye(3),atol=1e-12):raise ValueError('UNSUPPORTED_SHOULDER_REST')
    relative=(parent_rotation@chain.rest[0]).T@target_rotation
    primary=R.from_matrix(relative).as_euler('YXZ')
    alternate=np.array([primary[0]+np.pi,np.pi-primary[1],primary[2]+np.pi])
    candidates=[]
    for base in (primary,alternate):
        base=base.copy();base[1]-=restvec[0]
        for wraps in itertools.product((-1,0,1),repeat=3):
            angles=base+2*np.pi*np.array(wraps)
            if np.all(angles>=chain.bounds[:3,0]) and np.all(angles<=chain.bounds[:3,1]):
                if not any(np.max(abs(angles-other))<1e-6 for other in candidates):candidates.append(angles)
    return candidates


def choose_path(candidates,costs,step_weight=5.):
    if len(candidates)!=len(costs) or any(len(x)==0 for x in candidates):raise ValueError('EMPTY_BRANCH_FRAME')
    value=np.asarray(costs[0]);parents=[]
    for i in range(1,len(candidates)):
        transition=np.sum((candidates[i][:,None,:]-candidates[i-1][None,:,:])**2,axis=-1)*step_weight**2
        total=transition+value[None,:];back=total.argmin(axis=1)
        value=np.asarray(costs[i])+total[np.arange(len(back)),back];parents.append(back)
    path=[int(value.argmin())]
    for back in parents[::-1]:path.append(int(back[path[-1]]))
    path.reverse()
    return np.array([candidates[i][j] for i,j in enumerate(path)]),path,float(value.min())


def branch_path(chain,initial,parent_p,parent_r,target_p,target_r,weights,step_weight=5.):
    candidates=[];costs=[];no_exact=[]
    for i in range(len(initial)):
        def error(angles):
            p,r=chain.fk(angles[None],parent_p[i:i+1],parent_r[i:i+1])
            return (se3_error(p,r,target_p[i:i+1],target_r[i:i+1])*weights).ravel()
        options=[initial[i].copy()]
        branches=shoulder_branches(chain,parent_r[i],target_r[i,0])
        if not branches:no_exact.append(i)
        for branch in branches:
            pose=initial[i].copy();pose[:3]=branch
            def distal_error(x):
                q=pose.copy();q[3:]=x;return error(q)
            fit=least_squares(distal_error,np.clip(pose[3:],chain.bounds[3:,0]+1e-7,chain.bounds[3:,1]-1e-7),
                              bounds=(chain.bounds[3:,0]+1e-7,chain.bounds[3:,1]-1e-7),max_nfev=60,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            pose[3:]=fit.x;options.append(pose)
        candidates.append(np.array(options));costs.append(np.array([np.sum(error(q)**2) for q in options]))
    q,path,total=choose_path(candidates,costs,step_weight)
    return q,{'candidate_counts':[len(c) for c in candidates],'chosen_indices':path,
              'frames_without_exact_bounded_shoulder_target':no_exact,'whole_path_cost':total,'step_weight':step_weight,
              'reference_input':False,'method':'analytic bounded YXZ branches; distal fit; whole-clip dynamic programming'}
