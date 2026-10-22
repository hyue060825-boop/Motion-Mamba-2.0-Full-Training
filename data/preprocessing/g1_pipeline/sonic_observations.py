"""CPU-side SONIC G1-mode observations from audited deployment conventions.

This module only constructs policy inputs. It does not execute a controller,
create a robot connection, or grant dynamic QA acceptance.
"""
from collections import deque
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from .common import ROOT,sha


class SonicObservations:
    def __init__(self,audit_path=None):
        path=audit_path or ROOT/'audit/sonic_controller_static_interface.json'
        self.audit=json.loads(path.read_text())
        for name,expected in self.audit['source_hashes'].items():
            if sha(name)!=expected:raise ValueError('SONIC_INTERFACE_SOURCE_CHANGED:'+name)
        self.constants={name:np.asarray(v) for name,v in self.audit['compiled_deployment_constants'].items()}
        self.to_policy=self.constants['mujoco_to_isaaclab'].astype(int)
        self.to_mujoco=self.constants['isaaclab_to_mujoco'].astype(int)
        if not np.array_equal(self.to_policy[self.to_mujoco],np.arange(29)):raise ValueError('SONIC_ORDER_NOT_INVERTIBLE')
        self.encoder_slices={v['name']:slice(v['start'],v['stop']) for v in self.audit['encoder_layout']}
        self.decoder_slices={v['name']:slice(v['start'],v['stop']) for v in self.audit['decoder_layout']}
        mode=next(m for m in self.audit['encoder_modes'] if m['mode_id']==0)
        expected={'encoder_mode_4','motion_joint_positions_10frame_step5','motion_joint_velocities_10frame_step5','motion_anchor_orientation_10frame_step5'}
        if set(mode['required_observations'])!=expected:raise ValueError('SONIC_G1_MODE_CONTRACT_CHANGED')
        self.history=deque(maxlen=10)
        self.last_time=None

    def encoder(self,joint_positions,joint_velocities,reference_quaternions,frame_index,base_quaternion,heading_delta=0.,*,timestamps):
        positions=np.asarray(joint_positions,dtype=np.float64);velocities=np.asarray(joint_velocities,dtype=np.float64)
        quats=np.asarray(reference_quaternions,dtype=np.float64);base=np.asarray(base_quaternion,dtype=np.float64)
        n=len(positions)
        times=np.asarray(timestamps)
        if times.shape!=(n,) or not np.isfinite(times).all() or n<2 or not np.allclose(np.diff(times),.02,atol=1e-6,rtol=0):raise ValueError('SONIC_REFERENCE_MUST_BE_PHYSICAL_50HZ')
        if positions.shape!=(n,29) or velocities.shape!=(n,29) or quats.shape!=(n,4) or base.shape!=(4,):raise ValueError('SONIC_REFERENCE_SHAPE')
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<n:raise ValueError('SONIC_REFERENCE_INDEX')
        if not all(np.isfinite(x).all() for x in (positions,velocities,quats,base)):raise ValueError('SONIC_NONFINITE_INPUT')
        if np.max(abs(np.linalg.norm(quats,axis=1)-1))>1e-5 or abs(np.linalg.norm(base)-1)>1e-5:raise ValueError('SONIC_NONUNIT_QUATERNION')
        # Future policy context is clamped at the endpoint; physical replay length
        # remains the reference duration. The input grid must be verified at 50 Hz.
        indices=np.minimum(frame_index+np.arange(10)*5,n-1)
        out=np.zeros(self.audit['encoder_width'],dtype=np.float32)
        out[self.encoder_slices['encoder_mode_4']]=0 # mode ID=0 and three unused zeros
        out[self.encoder_slices['motion_joint_positions_10frame_step5']]=positions[indices][:,self.to_policy].ravel()
        out[self.encoder_slices['motion_joint_velocities_10frame_step5']]=velocities[indices][:,self.to_policy].ravel()
        relative=R.from_quat(base,scalar_first=True).inv()*R.from_euler('z',heading_delta)*R.from_quat(quats[indices],scalar_first=True)
        out[self.encoder_slices['motion_anchor_orientation_10frame_step5']]=relative.as_matrix()[:,:,:2].reshape(-1)
        return out[None]

    def push_state(self,joint_positions,joint_velocities,base_quaternion,base_angular_velocity_local,previous_action_policy,*,time_s):
        q=np.asarray(joint_positions);v=np.asarray(joint_velocities);quat=np.asarray(base_quaternion)
        omega=np.asarray(base_angular_velocity_local);action=np.asarray(previous_action_policy)
        if q.shape!=(29,) or v.shape!=(29,) or quat.shape!=(4,) or omega.shape!=(3,) or action.shape!=(29,):raise ValueError('SONIC_STATE_SHAPE')
        if not all(np.isfinite(x).all() for x in (q,v,quat,omega,action)):raise ValueError('SONIC_NONFINITE_STATE')
        if abs(np.linalg.norm(quat)-1)>1e-5:raise ValueError('SONIC_NONUNIT_STATE_QUATERNION')
        if not np.isfinite(time_s) or (self.last_time is not None and abs(time_s-self.last_time-.02)>1e-6):raise ValueError('SONIC_STATE_MUST_BE_PHYSICAL_50HZ')
        state={'q':(q-self.constants['default_angles'])[self.to_policy],
               'v':v[self.to_policy],'omega':omega.copy(),'action':action.copy(),
               'gravity':R.from_quat(quat,scalar_first=True).inv().apply([0.,0.,-1.])}
        if not self.history:
            # Explicit startup history protocol: pre-t=0 state held at first
            # observed state. No output motion frames are synthesized.
            for _ in range(10):self.history.append({k:x.copy() for k,x in state.items()})
        else:self.history.append(state)
        self.last_time=float(time_s)

    def decoder(self,token):
        token=np.asarray(token)
        if token.size!=64 or not np.isfinite(token).all():raise ValueError('SONIC_TOKEN_SHAPE')
        if len(self.history)!=10:raise ValueError('SONIC_HISTORY_NOT_INITIALIZED')
        out=np.zeros(self.audit['decoder_width'],dtype=np.float32)
        out[self.decoder_slices['token_state']]=token.reshape(-1)
        mapping={'his_base_angular_velocity_10frame_step1':'omega','his_body_joint_positions_10frame_step1':'q',
                 'his_body_joint_velocities_10frame_step1':'v','his_last_actions_10frame_step1':'action','his_gravity_dir_10frame_step1':'gravity'}
        for name,key in mapping.items():out[self.decoder_slices[name]]=np.array([x[key] for x in self.history]).ravel()
        return out[None]

    def joint_target(self,policy_action):
        action=np.asarray(policy_action)
        if action.size!=29 or not np.isfinite(action).all():raise ValueError('SONIC_INVALID_ACTION')
        return action.reshape(29)[self.to_mujoco]*self.constants['g1_action_scale']+self.constants['default_angles']
