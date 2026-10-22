import numpy as np
import pytest
from g1_pipeline.hiphi_native import read_bvh,world_fk


def test_six_channel_translations_and_end_sites_use_actual_local_transforms(tmp_path):
    path=tmp_path/'test.bvh'
    path.write_text('''HIERARCHY
ROOT Hips
{
 OFFSET 99 99 99
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Child
 {
  OFFSET 100 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  End Site
  {
   OFFSET 0 20 0
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.04
10 0 0 90 0 0 100 0 0 0 0 0
20 0 0 90 0 0 120 0 0 0 0 0
''')
    source=read_bvh(path)
    p,r=world_fk(source['local_positions_m'],source['local_rotations'],[n['parent'] for n in source['nodes']])
    # Channels are absolute local translations; adding OFFSET would double the arm.
    np.testing.assert_allclose(p[0],[[.1,0,0],[.1,1,0],[-.1,1,0]],atol=1e-12)
    np.testing.assert_allclose(p[1],[[.2,0,0],[.2,1.2,0],[0,1.2,0]],atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-12)
    assert source['nodes'][-1]['end_site']


def test_wrong_frame_count_is_rejected(tmp_path):
    path=tmp_path/'bad.bvh'
    path.write_text('HIERARCHY\nROOT Hips\n{\nOFFSET 0 0 0\nCHANNELS 3 Xrotation Yrotation Zrotation\n}\nMOTION\nFrames: 2\nFrame Time: 0.04\n0 0 0\n')
    with pytest.raises(ValueError,match='INVALID_BVH_MOTION_DATA'):read_bvh(path)
