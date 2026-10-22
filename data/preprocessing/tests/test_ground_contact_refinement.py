import mujoco
import numpy as np
from g1_pipeline.gmr_collision_refinement import collision_terms
from g1_pipeline.gmr_ground_contact_refinement import floor_pairs


def test_plane_penetration_gradient_matches_root_translation():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 .08"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);pairs=floor_pairs(model)
    residual,jac=collision_terms(model,data,pairs,minimum=-.015)
    np.testing.assert_allclose(residual,[.005],atol=1e-10)
    np.testing.assert_allclose(jac[0,:3],[0,0,-1],atol=1e-10)
    outputs=[]
    for sign in (-1,1):
        data.qpos[2]=.08+sign*1e-6;mujoco.mj_forward(model,data)
        outputs.append(collision_terms(model,data,pairs,minimum=-.015)[0])
    np.testing.assert_allclose((outputs[1]-outputs[0])/2e-6,jac[:,2],atol=1e-7)
