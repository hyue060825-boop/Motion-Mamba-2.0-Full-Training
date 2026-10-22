"""Source-specific static landmark/body-frame mapping; no motion IK here.

Native SMPL-H geometry is kept in its own joint axes. Robot anatomical rest
poses are the same fixed target poses used in the BONES rest audit, while
source offsets/scales are derived anew from each native rest shape.
"""
import copy
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,write_json,sha
from .robot import compile_target
from .hml_native import SMPLH_IDS
from .hml_native_runner import native_parameters

C=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
SMPL_MAP={'pelvis':'pelvis','torso_link':'spine3'}
for side in ('left','right'):
    for robot,human in [('hip_yaw','hip'),('knee','knee'),('ankle_roll','ankle'),
                        ('shoulder_yaw','shoulder'),('elbow','elbow'),('wrist_yaw','wrist')]:
        SMPL_MAP[f'{side}_{robot}_link']=f'{side}_{human}'


def target_anatomical_rest(model):
    q=model.qpos0.copy();q[3:7]=R.from_euler('z',-np.pi/2).as_quat(scalar_first=True)
    for side,sign in [('left',1),('right',-1)]:
        q[model.joint(side+'_shoulder_roll_joint').qposadr[0]]=sign*np.pi/2
        q[model.joint(side+'_elbow_joint').qposadr[0]]=np.pi/2
    if np.any(q[7:]<model.jnt_range[1:,0]-1e-4) or np.any(q[7:]>model.jnt_range[1:,1]+1e-4):raise ValueError('ANATOMICAL_REST_LIMIT_FAILURE')
    d=mujoco.MjData(model);d.qpos[:]=q;mujoco.mj_forward(model,d)
    return q,d


def build_mapping(rest_positions,rest_rotations,mapping,source_pelvis_height,source_height,*,source_adapter,root_name,provenance):
    m,asset=compile_target();q,d=target_anatomical_rest(m)
    if not .4<source_pelvis_height<1.5 or not .8<source_height<2.5:raise ValueError('IMPLAUSIBLE_SOURCE_REST_SIZE')
    scale=float(q[2]/source_pelvis_height)
    template=ROOT/'configs/bones_temporal_arms_v15.json';cfg=copy.deepcopy(json.loads(template.read_text()))
    cfg['source_adapter']=source_adapter;cfg['human_root_name']=root_name
    cfg['human_height_assumption']=float(source_height)
    cfg['human_scale_table']={name:scale for name in mapping.values()}
    records=[]
    for body,name in mapping.items():
        bid=m.body(body).id;p=np.asarray(rest_positions[name]);r=np.asarray(rest_rotations[name])
        robot_r=d.xmat[bid].reshape(3,3);offset_r=r.T@robot_r
        offset_p=robot_r.T@(d.xpos[bid]-scale*p)
        if not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1):raise ValueError('INVALID_NATIVE_REST_ROTATION')
        for table in ('ik_match_table1','ik_match_table2'):
            old=cfg[table][body];cfg[table][body]=[name,old[1],old[2],offset_p.tolist(),R.from_matrix(offset_r).as_quat(scalar_first=True).tolist()]
        # Independent algebraic rest invariant; no numerical optimization and
        # no paired robot motion are used to select mapping parameters.
        actual_r=r@offset_r;actual_p=scale*p+actual_r@offset_p
        records.append({'source':name,'target':body,'source_rest_position_m':p.tolist(),'source_rest_rotation':r.tolist(),
            'target_rest_position_m':d.xpos[bid].tolist(),'target_rest_rotation':robot_r.tolist(),
            'rest_position_residual_m':float(np.linalg.norm(actual_p-d.xpos[bid])),
            'rest_rotation_residual_deg':float(np.rad2deg((R.from_matrix(actual_r).inv()*R.from_matrix(robot_r)).magnitude()))})
    cfg['scale_strategy']={'name':'native_rest_pelvis_height_to_locked_target_rest', 'ratio':scale,
        'source_pelvis_height_m':source_pelvis_height,'source_height_m':source_height,
        'actual_human_height_argument':None,'reason':'ratio already applied exactly once in human_scale_table; no second actor-height scaling'}
    audit={'status':'STATIC_MAPPING_ONLY_PENDING_ROBOT_SMOKE','source_adapter':source_adapter,'asset_hash':asset['asset_hash'],
        'weights_and_temporal_template_sha256':sha(template),'target_rest_qpos':q.tolist(),'scale_strategy':cfg['scale_strategy'],
        'mapping':records,'source_provenance':provenance,'source_motion_reference_used':False}
    return cfg,audit


def smplh_mapping(npz_path,*,mirror=False):
    params=native_parameters(npz_path,start_frame=0,end_frame=1);g=params['geometry'];betas=params['betas']
    rest=g['rest']+np.einsum('jck,k->jc',g['joint_shape'],betas)
    vertices=g['template']+np.einsum('vck,k->vc',g['shape'],betas)
    floor=float(vertices[:,1].min());height=float(np.ptp(vertices[:,1]));pelvis_height=float(rest[0,1]-floor)
    canonical=(rest-rest[0])@C.T+np.array([0,0,pelvis_height])
    positions={name:canonical[idx].copy() for name,idx in SMPLH_IDS.items()};rotations={name:C.copy() for name in SMPLH_IDS}
    if mirror:
        reflect=np.diag([-1.,1,1]);old=positions.copy()
        for name in positions:
            opposite=name.replace('left_','right_') if name.startswith('left_') else name.replace('right_','left_') if name.startswith('right_') else name
            positions[name]=reflect@old[opposite]
    cfg,audit=build_mapping(positions,rotations,SMPL_MAP,pelvis_height,height,
        source_adapter='native_fk_to_gmr_smplx_semantics',root_name='pelvis',
        provenance={'native_path':str(npz_path),'native_sha256':sha(npz_path),'native_model':g['model_path'],
                    'native_model_sha256':sha(g['model_path']),'dmpl_model_sha256':sha(g['dmpl_path']),
                    'body_model_type':'smplh','pose_layout':156,'gender':params['gender'],'betas':betas.tolist(),
                    'canonical_floor':'minimum canonical shaped mesh Y; DMPL zero','mirror':mirror,
                    'dynamic_input_basis':'unchanged native AMASS world; source local joint axes retained'})
    return cfg,audit


def mirror_frames(frames):
    """HumanML3D mirror in native world X, with semantic sides exchanged."""
    s=np.diag([-1.,1,1]);result=[]
    for frame in frames:
        value={}
        for name in frame:
            other=name.replace('left_','right_') if name.startswith('left_') else name.replace('right_','left_') if name.startswith('right_') else name
            p,q=frame[other];value[name]=(s@p,R.from_matrix(s@R.from_quat(q,scalar_first=True).as_matrix()@s).as_quat(scalar_first=True))
        result.append(value)
    return result
