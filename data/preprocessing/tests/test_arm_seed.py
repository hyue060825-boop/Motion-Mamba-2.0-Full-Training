import numpy as np
import pytest
from g1_pipeline.common import ROOT
from g1_pipeline.locked_target import create_retargeter
from g1_pipeline.gmr_arm_seed import arm_seed


def problem():
    solver=create_retargeter(ROOT/'configs/bones_soma_rootweight_v6.json',ROOT/'assets/project_g1_v1/g1.xml')
    q=solver.model.qpos0.copy()
    for side in ('left','right'):
        for name,value in [('shoulder_pitch',.5),('shoulder_roll',.25 if side=='left' else -.25),('shoulder_yaw',.4),('elbow',1.)]:
            q[solver.model.joint(f'{side}_{name}_joint').qposadr[0]]=value
    solver.configuration.update(q)
    for body,entry in solver.ik_match_table2.items():
        solver.human_body_to_task2[entry[0]].set_target(solver.configuration.get_transform_frame_to_world(body,'body'))
    return solver,q


def test_exact_robot_target_improves_with_bounded_arm_only_seed():
    solver,target=problem()
    adr=[solver.model.joint(f'left_{n}_joint').qposadr[0] for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
    initial=target.copy();initial[adr]=0.;solver.configuration.update(initial)
    pose,info=arm_seed(solver,'left')
    assert info['best_arm_cost']<1e-8
    assert info['best_arm_cost']<info['initial_arm_cost']
    np.testing.assert_array_equal(pose[np.setdiff1d(np.arange(36),adr)],initial[np.setdiff1d(np.arange(36),adr)])
    assert np.all(pose[7:]>=solver.model.jnt_range[1:,0])
    assert np.all(pose[7:]<=solver.model.jnt_range[1:,1])


def test_arm_seed_exception_restores_configuration(monkeypatch):
    solver,target=problem()
    def fail(*args,**kwargs):
        solver.configuration.update(solver.model.qpos0.copy())
        raise RuntimeError('test optimization failure')
    monkeypatch.setattr('g1_pipeline.gmr_arm_seed.least_squares',fail)
    with pytest.raises(RuntimeError):arm_seed(solver,'left')
    np.testing.assert_array_equal(solver.configuration.q,target)
