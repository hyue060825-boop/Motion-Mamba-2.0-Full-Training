import numpy as np
from scipy.spatial.transform import Rotation
from g1_pipeline.gmr_soma import parse


def test_local_slerp_before_fk_preserves_rotating_bone(tmp_path):
    # Parent rotates 90 degrees while translating 10 cm. At 40 ms the child
    # must lie on a unit circle, not the chord from interpolated global points.
    path=tmp_path/'analytic.bvh'
    header='''HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 JOINT Child
 {
  OFFSET 100 0 0
  CHANNELS 3 Yrotation Xrotation Zrotation
  End Site
  {
   OFFSET 1 0 0
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.08
'''
    path.write_text(header+'0 0 0 0 0 0 0 0 0\n10 0 0 90 0 0 0 0 0\n')
    frames,fps=parse(path,{'Hips':'Hips','Child':'Child'},target_timestamps=[0,.04,.08],rotation_basis='world_only')
    c=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    assert fps==12.5
    for i,angle in enumerate([0,45,90]):
        root=np.array([.05*i,0,0]);r=Rotation.from_euler('y',angle,degrees=True)
        np.testing.assert_allclose(frames[i]['Hips'][0],c@root,atol=1e-12)
        np.testing.assert_allclose(frames[i]['Child'][0],c@(root+r.apply([1,0,0])),atol=1e-12)
        actual=Rotation.from_quat(frames[i]['Child'][1],scalar_first=True).as_matrix()
        np.testing.assert_allclose(actual,c@r.as_matrix(),atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(frames[i]['Child'][0]-frames[i]['Hips'][0]),1,atol=1e-12)
