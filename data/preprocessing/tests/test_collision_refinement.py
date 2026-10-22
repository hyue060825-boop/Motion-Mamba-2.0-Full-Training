import mujoco
import numpy as np
from g1_pipeline.gmr_collision_refinement import collision_terms


def test_penetration_jacobian_spheres_matches_physical_separation():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><geom type="sphere" size="0.1"/></body>
      <body pos="0.15 0 0"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.1" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    residual, jac = collision_terms(model, data, [(0, 1)])
    np.testing.assert_allclose(residual, [.048], atol=1e-10)
    np.testing.assert_allclose(jac, [[-1]], atol=1e-10)
    eps = 1e-6; values = []
    for x in [-eps, eps]:
        data.qpos[0] = x; mujoco.mj_forward(model, data)
        values.append(collision_terms(model, data, [(0, 1)])[0])
    np.testing.assert_allclose((values[1]-values[0])/(2*eps), jac[:, 0], atol=1e-7)
    data.qpos[0] = .1; mujoco.mj_forward(model, data)
    residual, jac = collision_terms(model, data, [(0, 1)])
    assert np.count_nonzero(residual) == np.count_nonzero(jac) == 0
