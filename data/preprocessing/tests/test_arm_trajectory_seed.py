import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT
from g1_pipeline.locked_target import create_retargeter
from g1_pipeline.gmr_arm_trajectory_seed import ArmChain,se3_error,optimize


def test_vector_arm_fk_and_se3_error_against_mujoco_and_mink():
    solver=create_retargeter(ROOT/'configs/bones_soma_rootweight_v6.json',ROOT/'assets/project_g1_v1/g1.xml')
    m=solver.model;d=mujoco.MjData(m);rng=np.random.default_rng(82)
    for side in ('left','right'):
        chain=ArmChain(m,side)
        for _ in range(12):
            q=m.qpos0.copy();q[:3]=rng.normal(size=3);q[3:7]=R.random(random_state=rng).as_quat(scalar_first=True)
            q[7:]=rng.uniform(m.jnt_range[1:,0],m.jnt_range[1:,1])
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            p,r=chain.fk(q[chain.adr][None],d.xpos[chain.parent][None],d.xmat[chain.parent].reshape(1,3,3))
            np.testing.assert_allclose(p[0],d.xpos[chain.bodies[[2,3,6]]],atol=1e-12)
            np.testing.assert_allclose(r[0],d.xmat[chain.bodies[[2,3,6]]].reshape(3,3,3),atol=1e-12)
            solver.configuration.update(q)
            import mink
            tp=rng.normal(size=(1,3,3));tr=R.random(3,random_state=rng).as_matrix()[None]
            error=se3_error(p,r,tp,tr)
            for j,part in enumerate(('shoulder_yaw','elbow','wrist_yaw')):
                body=f'{side}_{part}_link'
                task=solver.human_body_to_task2[solver.ik_match_table2[body][0]]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(R.from_matrix(tr[0,j]).as_quat(scalar_first=True)),tp[0,j]))
                np.testing.assert_allclose(error[0,j],task.compute_error(solver.configuration),atol=1e-10)


def test_constant_known_arm_trajectory_is_unchanged():
    solver=create_retargeter(ROOT/'configs/bones_soma_rootweight_v6.json',ROOT/'assets/project_g1_v1/g1.xml')
    chain=ArmChain(solver.model,'left');n=5
    angles=np.tile(np.mean(chain.bounds,axis=1),(n,1));p=np.zeros((n,3));r=np.tile(np.eye(3),(n,1,1))
    tp,tr=chain.fk(angles,p,r)
    out,info=optimize(chain,angles,p,r,tp,tr,np.ones((3,6)))
    np.testing.assert_allclose(out,angles,atol=1e-10)
    assert info['final_cost']<1e-20
