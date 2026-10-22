import numpy as np
from scipy.spatial.transform import Rotation as R
from g1_pipeline.motion_duplicates import physical_descriptor


def test_descriptor_ignores_fixed_world_heading_xy_but_not_height_or_hinges():
    q=np.zeros((50,36));q[:,2]=.8;q[:,0]=np.arange(50)*.02;q[:,3]=1;q[:,10]=np.sin(np.arange(50)*.1)*.2
    altered=q.copy();turn=R.from_euler('z',1.2);altered[:,:3]=turn.apply(q[:,:3])+[5,8,0]
    altered[:,3:7]=(turn*R.from_quat(q[:,3:7],scalar_first=True)).as_quat(scalar_first=True)
    a,_=physical_descriptor(q);b,_=physical_descriptor(altered);np.testing.assert_allclose(a,b,atol=1e-12)
    altered[:,2]+=.2;c,_=physical_descriptor(altered);assert not np.allclose(a,c)
