import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT
from g1_pipeline.sonic_scene import prepare_scene,torque_control


def test_actuated_scene_preserves_exact_fk_and_effort_caps():
    model,report=prepare_scene();base=mujoco.MjModel.from_xml_path(str(ROOT/'assets/project_g1_v1/g1.xml'))
    a=mujoco.MjData(base);b=mujoco.MjData(model);rng=np.random.default_rng(713)
    assert model.nu==29 and all(report['checks'].values())
    for _ in range(12):
        q=base.qpos0.copy();q[:3]=rng.normal(size=3);q[3:7]=R.random(random_state=rng).as_quat(scalar_first=True)
        q[7:]=rng.uniform(base.jnt_range[1:,0],base.jnt_range[1:,1]);a.qpos[:]=q;b.qpos[:]=q
        mujoco.mj_forward(base,a);mujoco.mj_forward(model,b)
        np.testing.assert_array_equal(a.xpos,b.xpos);np.testing.assert_array_equal(a.xmat,b.xmat)
    constants=json.loads((ROOT/'audit/sonic_controller_static_interface.json').read_text())['compiled_deployment_constants']
    q=base.qpos0.copy();v=np.zeros(35)
    torque,info=torque_control(model,q,v,q[7:],constants)
    np.testing.assert_array_equal(torque,np.zeros(29));assert not info['saturated'].any()
    torque,info=torque_control(model,q,v,q[7:]+100,constants)
    np.testing.assert_array_equal(torque,base.jnt_actfrcrange[1:,1]);assert info['saturated'].all()
    # A policy for newer hip motors must not silently increase this target's cap.
    assert torque[0]==88 and torque[6]==88
