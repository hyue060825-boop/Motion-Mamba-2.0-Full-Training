"""Measured torso inclination for bow completion, independent of leg motion."""
from motion_mamba2.paths import artifact_path
import mujoco
import numpy as np
_model=None
def torso_pitch(q):
    global _model
    if _model is None:_model=mujoco.MjModel.from_xml_path(artifact_path('scene_29dof_xml_b80ed52e2a'))
    data=mujoco.MjData(_model);body=mujoco.mj_name2id(_model,mujoco.mjtObj.mjOBJ_BODY,'torso_link');assert body>=0
    result=[]
    for frame in q:
        data.qpos[:]=frame;mujoco.mj_forward(_model,data);forward=data.xmat[body].reshape(3,3)[:,0]
        result.append(np.rad2deg(np.arctan2(-forward[2],np.linalg.norm(forward[:2]))))
    return np.array(result)
