import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT
from g1_pipeline.locked_target import create_retargeter


def test_gmr_static_targets_match_locked_robot_and_preserve_palm_tilt():
    report=json.loads((ROOT/'audit/soma_rest_mapping_v3.json').read_text())
    g=create_retargeter(ROOT/'configs/bones_soma_rest_v3.json',ROOT/'assets/project_g1_v1/g1.xml')
    q=np.asarray(report['target_tpose_qpos'])
    assert np.all(q[7:]>=g.model.jnt_range[1:,0])
    assert np.all(q[7:]<=g.model.jnt_range[1:,1])
    d=mujoco.MjData(g.model);d.qpos[:]=q;mujoco.mj_forward(g.model,d)
    human={r['source']:[np.asarray(r['source_rest_position_m']),R.from_matrix(r['source_rest_rotation']).as_quat(scalar_first=True)] for r in report['mapping']}
    g.update_targets(human,offset_to_ground=False)
    for row in report['mapping']:
        p,quat=g.scaled_human_data[row['source']];body=g.model.body(row['target_body']).id
        np.testing.assert_allclose(p,d.xpos[body],atol=1e-7)
        actual=R.from_quat(quat,scalar_first=True)
        if not row['source'].endswith('Hand'):
            np.testing.assert_allclose(actual.as_matrix(),d.xmat[body].reshape(3,3),atol=1e-7)
        else:
            tilt=np.rad2deg((R.from_matrix(d.xmat[body].reshape(3,3)).inv()*actual).magnitude())
            # Native rest asset has a nonzero palm tilt; normalizing it away is wrong.
            assert 9<tilt<11


def test_root_initialization_removes_world_heading_dependence():
    from g1_pipeline.locked_target import initialize_root_from_human
    report=json.loads((ROOT/'audit/soma_rest_mapping_v3.json').read_text())
    config=ROOT/'configs/bones_soma_rest_v4.json';xml=ROOT/'assets/project_g1_v1/g1.xml'
    turns=[R.identity(),R.from_euler('z',173,degrees=True)];outputs=[]
    for turn in turns:
        g=create_retargeter(config,xml)
        human={row['source']:[turn.apply(row['source_rest_position_m']),
               (turn*R.from_matrix(row['source_rest_rotation'])).as_quat(scalar_first=True)] for row in report['mapping']}
        initialize_root_from_human(g,human)
        outputs.append(g.retarget(human).copy())
    a,b=outputs
    np.testing.assert_allclose(a[7:],b[7:],atol=1e-5,rtol=0)
    np.testing.assert_allclose(turns[1].apply(a[:3]),b[:3],atol=1e-6,rtol=0)
    angle=(R.from_quat(b[3:7],scalar_first=True).inv()*turns[1]*R.from_quat(a[3:7],scalar_first=True)).magnitude()
    assert angle<1e-5
