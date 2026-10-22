"""Actuated copy of the exact locked G1 geometry for future SONIC QA.

Only torque motors are added. Native inertia, contact geometry, joint limits,
armature and effort ranges are retained; no controller compatibility is claimed
by constructing this scene.
"""
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .common import ROOT,sha,write_json,atomic_bytes


def prepare_scene():
    source=ROOT/'assets/project_g1_v1/g1.xml'
    base=mujoco.MjModel.from_xml_path(str(source))
    if (base.nq,base.nv,base.nu)!=(36,35,0):raise ValueError('UNEXPECTED_LOCKED_TARGET')
    tree=ET.parse(source);xml=tree.getroot()
    if xml.find('actuator') is not None:raise ValueError('LOCKED_ASSET_ALREADY_HAS_ACTUATORS')
    actuator=ET.SubElement(xml,'actuator')
    for j in range(1,base.njnt):
        if not base.jnt_actfrclimited[j]:raise ValueError('MISSING_LOCKED_JOINT_EFFORT_RANGE')
        lower,upper=base.jnt_actfrcrange[j]
        ET.SubElement(actuator,'motor',name=base.joint(j).name+'_motor',joint=base.joint(j).name,gear='1',
                      ctrllimited='true',ctrlrange=f'{lower:.17g} {upper:.17g}',forcelimited='true',forcerange=f'{lower:.17g} {upper:.17g}')
    path=ROOT/'assets/project_g1_sonic_scene_v1/g1.xml';atomic_bytes(path,ET.tostring(xml,encoding='utf-8'))
    model=mujoco.MjModel.from_xml_path(str(path))
    attrs=['jnt_type','jnt_qposadr','jnt_dofadr','jnt_axis','jnt_pos','jnt_range','jnt_actfrcrange','jnt_actfrclimited',
           'qpos0','dof_armature','dof_damping','dof_frictionloss','body_parentid','body_pos','body_quat','body_mass',
           'body_inertia','geom_bodyid','geom_type','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','mesh_vert','mesh_face']
    checks={name:bool(np.array_equal(getattr(model,name),getattr(base,name))) for name in attrs}
    checks['motor_order']=bool(np.array_equal(model.actuator_trnid[:,0],np.arange(1,base.njnt)))
    checks['timestep']=model.opt.timestep==base.opt.timestep
    if not all(checks.values()) or (model.nq,model.nv,model.nu)!=(36,35,29):raise ValueError('ACTUATED_SCENE_MODIFIED_LOCKED_GEOMETRY')
    params=json.loads((ROOT/'audit/sonic_controller_static_interface.json').read_text())['compiled_deployment_constants']
    report={'status':'ACTUATED_SCENE_CONTRACT_PASS','source_xml_sha256':sha(source),'actuated_xml_sha256':sha(path),
            'path':str(path),'nq':model.nq,'nv':model.nv,'nu':model.nu,'checks':checks,'timestep_s':model.opt.timestep,
            'control_dt_s':.02,'kp_source':'compiled SONIC deployment policy_parameters.hpp',
            'kp':params['kps'],'kd':params['kds'],'effort_limits':model.actuator_ctrlrange.tolist(),
            'known_mismatch':'SONIC training uses newer 139 Nm hip-pitch motors; locked target retains its original 88 Nm hip-pitch effort range and armature.',
            'controller_tracking_validated':False,'dynamic_qa_status':'NOT_RUN'}
    write_json(ROOT/'audit/sonic_actuated_scene.json',report)
    return model,report


def torque_control(model,qpos,qvel,target,constants):
    qpos=np.asarray(qpos);qvel=np.asarray(qvel);target=np.asarray(target)
    if qpos.shape!=(36,) or qvel.shape!=(35,) or target.shape!=(29,) or not all(np.isfinite(v).all() for v in (qpos,qvel,target)):raise ValueError('INVALID_SONIC_CONTROL_STATE')
    raw=np.asarray(constants['kps'])*(target-qpos[7:])-np.asarray(constants['kds'])*qvel[6:]
    torque=np.clip(raw,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
    return torque,{'saturated':np.abs(raw-torque)>1e-8,'requested_torque':raw}
