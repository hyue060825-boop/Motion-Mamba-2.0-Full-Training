"""Register GMR against the locked project robot, with only mesh paths resolved.

IK does not require actuators. This is a kinematic target, not a controller scene.
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .common import ROOT, XML, MESHES, atomic_bytes, write_json, sha, digest
from .robot import compile_target

ASSET_DIR = ROOT / 'assets/project_g1_v1'

def prepare():
    reference, provenance = compile_target()
    tree = ET.parse(XML)
    tree.find('compiler').set('meshdir', str(MESHES))
    content = ET.tostring(tree.getroot(), encoding='utf-8')
    path = ASSET_DIR / 'g1.xml'
    atomic_bytes(path, content)
    model = mujoco.MjModel.from_xml_path(str(path))
    checks = {}
    for attr in ('jnt_type','jnt_qposadr','jnt_dofadr','jnt_axis','jnt_pos','jnt_range',
                 'jnt_limited','qpos0','body_parentid','body_pos','body_quat','body_mass',
                 'body_inertia','geom_bodyid','geom_type','geom_pos','geom_quat','geom_size',
                 'geom_contype','geom_conaffinity','mesh_vert','mesh_face'):
        checks[attr] = bool(np.array_equal(getattr(reference,attr),getattr(model,attr)))
    checks['names'] = reference.names == model.names
    checks['timestep'] = reference.opt.timestep == model.opt.timestep
    rng = np.random.default_rng(20260915)
    poses = [model.qpos0.copy()]
    for j in range(29):
        for delta in (-.1, .1):
            q = model.qpos0.copy(); q[7+j] = np.clip(delta,*model.jnt_range[1+j]); poses.append(q)
    for _ in range(64):
        q = model.qpos0.copy(); q[:3] = rng.normal(size=3)
        q[3:7] = Rotation.random(random_state=rng).as_quat(scalar_first=True)
        q[7:] = rng.uniform(model.jnt_range[1:,0],model.jnt_range[1:,1]); poses.append(q)
    da,db = mujoco.MjData(reference),mujoco.MjData(model)
    pos_error=rot_error=0.
    for q in poses:
        da.qpos[:]=q; db.qpos[:]=q
        mujoco.mj_forward(reference,da); mujoco.mj_forward(model,db)
        pos_error=max(pos_error,float(np.max(np.abs(da.xpos-db.xpos))))
        rot_error=max(rot_error,float(np.max(np.abs(da.xmat-db.xmat))))
    report={'time':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'status':'PASS' if all(checks.values()) and pos_error<=1e-12 and rot_error<=1e-12 else 'FAIL',
        'scope':'resolved locked asset equality only; source adapter and calibration remain unvalidated',
        'source':provenance,'resolved_xml':str(path),'resolved_xml_sha256':sha(path),
        'nq':model.nq,'nv':model.nv,'nu':model.nu,'checks':checks,'tested_poses':len(poses),
        'fk_max_position_m':pos_error,'fk_max_rotation_matrix_abs':rot_error,
        'asset_hash':digest({'source':provenance,'resolved_xml':sha(path)})}
    write_json(ROOT/'audit/locked_target_equivalence.json',report)
    if report['status']!='PASS': raise ValueError('LOCKED_ASSET_MISMATCH')
    return path, report

def create_retargeter(config_path, xml_path=None, *, source_name='bvh_bones_soma', actual_human_height=None):
    sys.path.insert(0,str(ROOT/'deps/GMR'))
    from general_motion_retargeting import params, GeneralMotionRetargeting
    if xml_path is None: xml_path,_=prepare()
    params.ROBOT_XML_DICT['project_g1_v1']=Path(xml_path)
    params.IK_CONFIG_DICT.setdefault(source_name,{})['project_g1_v1']=Path(config_path)
    # New instance per clip prevents IK/temporal state crossing sample boundaries.
    return GeneralMotionRetargeting(source_name,'project_g1_v1',actual_human_height=actual_human_height,solver='daqp',verbose=False)


def initialize_root_from_human(retargeter, human_frame):
    """Use the first corrected human pelvis target, never the paired robot reference.

    This removes world-heading-dependent IK basins. Hinges keep the target's
    fixed neutral initialization and each clip still owns an independent solver.
    """
    import copy
    retargeter.update_targets(copy.deepcopy(human_frame),offset_to_ground=False)
    position,quaternion=retargeter.scaled_human_data[retargeter.human_root_name]
    q=retargeter.model.qpos0.copy()
    q[:3]=position;q[3:7]=quaternion/np.linalg.norm(quaternion)
    retargeter.configuration.update(q)

if __name__=='__main__':
    _,report=prepare(); print({k:report[k] for k in ('status','nq','nv','nu','tested_poses','fk_max_position_m')})
