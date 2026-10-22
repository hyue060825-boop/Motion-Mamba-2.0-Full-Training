import numpy as np
from scipy.spatial.transform import Rotation
from g1_pipeline.robot import compile_target
from g1_pipeline.bones_continuity_qa import metrics


def test_mujoco_root_velocity_ignores_quaternion_sign_and_wrap():
    model,_=compile_target();t=np.arange(8)*.04
    q=np.tile(model.qpos0,(len(t),1))
    q[:,:3]+=t[:,None]*[.2,-.1,.05]
    q[:,3:7]=Rotation.from_euler('z',(3.12+1.5*t)[:,None]).as_quat(scalar_first=True)
    q[::2,3:7]*=-1
    q[:,7]=.4*t
    result=metrics(model,q,t)
    np.testing.assert_allclose(result['root_angular_speed_p95_rad_s'],1.5,atol=1e-10)
    np.testing.assert_allclose(result['root_translation_speed_p95_m_s'],np.linalg.norm([.2,-.1,.05]),atol=1e-10)
    np.testing.assert_allclose(result['hinge_speed_p95_rad_s'][0],.4,atol=1e-10)
    assert max(result['hinge_acceleration_max_rad_s2'])<1e-10


def test_continuity_cli_uses_requested_namespace(monkeypatch):
    from g1_pipeline import bones_continuity_qa
    called=[]
    monkeypatch.setattr(bones_continuity_qa,'run',lambda namespace:called.append(namespace))
    bones_continuity_qa.main(['--namespace','bones_soma_rootweight_dev_v6'])
    assert called==['bones_soma_rootweight_dev_v6']
