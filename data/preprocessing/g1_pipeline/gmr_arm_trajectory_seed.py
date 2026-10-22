"""Offline bounded arm warm starts using GMR's SE(3) objective and time priors.

Only seven hinges per arm are searched. All other poses are fixed, and no
reference robot data enters the optimizer. These are experimental seeds,
not production retargeted trajectories.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix


SUFFIXES=('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')


class ArmChain:
    def __init__(self,model,side):
        self.ids=np.array([model.joint(f'{side}_{name}_joint').id for name in SUFFIXES])
        self.adr=model.jnt_qposadr[self.ids]
        self.bodies=model.jnt_bodyid[self.ids]
        if not np.array_equal(model.body_parentid[self.bodies[1:]],self.bodies[:-1]):
            raise ValueError('ARM_MUST_BE_SERIAL_CHAIN')
        if np.any(model.jnt_type[self.ids]!=3):raise ValueError('ARM_MUST_CONTAIN_ONLY_HINGES')
        self.parent=int(model.body_parentid[self.bodies[0]])
        self.pos=model.body_pos[self.bodies].copy()
        self.rest=R.from_quat(model.body_quat[self.bodies],scalar_first=True).as_matrix()
        self.axes=model.jnt_axis[self.ids].copy()
        self.jointpos=model.jnt_pos[self.ids].copy()
        self.bounds=model.jnt_range[self.ids].copy()

    def fk(self,angles,parent_position,parent_rotation):
        p=parent_position.copy();r=parent_rotation.copy();positions=[];rotations=[]
        for j in range(7):
            body_p=p+np.einsum('nij,j->ni',r,self.pos[j])
            body_r=r@self.rest[j]
            hinge=R.from_rotvec(angles[:,j,None]*self.axes[j]).as_matrix()
            p=body_p+np.einsum('nij,nj->ni',body_r,self.jointpos[j]-np.einsum('nij,j->ni',hinge,self.jointpos[j]))
            r=body_r@hinge
            if j in (2,3,6):positions.append(p);rotations.append(r)
        return np.stack(positions,axis=1),np.stack(rotations,axis=1)


def se3_error(position,rotation,target_position,target_rotation):
    """Batched log(current^-1 target), matching Mink FrameTask's body twist."""
    inverse=np.swapaxes(rotation,-1,-2)
    omega=R.from_matrix((inverse@target_rotation).reshape(-1,3,3)).as_rotvec().reshape(position.shape)
    translation=np.einsum('...ij,...j->...i',inverse,target_position-position)
    theta2=np.sum(omega*omega,axis=-1)
    theta=np.sqrt(theta2)
    coefficient=np.full_like(theta,1/12)
    use=theta2>1e-8
    coefficient[use]=(1-.5*theta[use]/np.tan(.5*theta[use]))/theta2[use]
    cross=np.cross(omega,translation)
    velocity=translation-.5*cross+coefficient[...,None]*np.cross(omega,cross)
    return np.concatenate([velocity,omega],axis=-1)


def optimize(chain,initial,parent_position,parent_rotation,target_position,target_rotation,weights,
             *,velocity_weight=.2,acceleration_weight=.004,dt=.04,max_nfev=120):
    n=len(initial)
    if n<3 or initial.shape!=(n,7):raise ValueError('NEED_FULL_ARM_SEQUENCE')
    def residual(flat):
        angles=flat.reshape(n,7)
        p,r=chain.fk(angles,parent_position,parent_rotation)
        task=se3_error(p,r,target_position,target_rotation)*weights
        return np.concatenate([task.ravel(),(np.diff(angles,axis=0)*velocity_weight/dt).ravel(),
                               (np.diff(angles,n=2,axis=0)*acceleration_weight/dt**2).ravel()])
    sparsity=lil_matrix((n*18+(n-1)*7+(n-2)*7,n*7),dtype=np.int8)
    for i in range(n):sparsity[i*18:(i+1)*18,i*7:(i+1)*7]=1
    offset=n*18
    for i in range(n-1):
        for j in range(7):sparsity[offset+i*7+j,[i*7+j,(i+1)*7+j]]=1
    offset+=(n-1)*7
    for i in range(n-2):
        for j in range(7):sparsity[offset+i*7+j,[i*7+j,(i+1)*7+j,(i+2)*7+j]]=1
    lower=np.tile(chain.bounds[:,0]+1e-7,n);upper=np.tile(chain.bounds[:,1]-1e-7,n)
    x=np.clip(initial.ravel(),lower,upper);before=residual(x)
    fit=least_squares(residual,x,bounds=(lower,upper),jac_sparsity=sparsity.tocsr(),
                      max_nfev=max_nfev,ftol=1e-5,xtol=1e-6,gtol=1e-5)
    return fit.x.reshape(n,7),{'initial_cost':float(before@before),'final_cost':float(fit.fun@fit.fun),
                             'nfev':fit.nfev,'optimality':float(fit.optimality),'solver_success':bool(fit.success),
                             'velocity_weight':velocity_weight,'acceleration_weight':acceleration_weight,
                             'physical_dt_s':dt,'stage':'ARM_TRAJECTORY_WARM_START_ONLY'}
