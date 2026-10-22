from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from g1_pipeline.common import ROOT
import sys
sys.path.insert(0,str(ROOT/'deps/GMR'))
import mink
from g1_pipeline.gmr_weighted_convergence import converge


def problem():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="arm">
        <joint name="hinge" axis="0 0 1" range="-60 60"/>
        <geom type="capsule" size=".02 .3"/>
        </body></worldbody></mujoco>''')
    c=mink.Configuration(model)
    task=mink.FrameTask('arm','body',position_cost=0.,orientation_cost=10.,lm_damping=1.)
    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(Rotation.from_euler('z',.4).as_quat(scalar_first=True)),np.zeros(3)))
    solver=SimpleNamespace(configuration=c,model=model,solver='daqp',damping=.5,ik_limits=[mink.ConfigurationLimit(model)])
    return solver,task


def test_known_hinge_target_and_weighted_descent():
    solver,task=problem();r=converge(solver,[task])
    np.testing.assert_allclose(solver.configuration.q,[.4],atol=1e-5)
    assert np.all(np.diff(r['cost_history'])<=0)
    assert r['final_cost']<1e-8


def test_failed_descent_restores_original_configuration(monkeypatch):
    solver,task=problem();q=solver.configuration.q.copy()
    monkeypatch.setattr(mink,'solve_ik',lambda *args:np.array([-1.]))
    r=converge(solver,[task])
    assert r['reason']=='no_descent'
    np.testing.assert_array_equal(q,solver.configuration.q)
    assert r['initial_cost']==r['final_cost']
