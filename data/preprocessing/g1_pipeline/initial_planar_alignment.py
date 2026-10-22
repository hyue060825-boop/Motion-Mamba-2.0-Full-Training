"""One initial planar transform using the full initial root orientation.

Development alternative to projecting a nearly vertical forward axis. No frame
other than frame zero contributes to the transform; root height is preserved.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def initial_alignment(candidate_first, reference_first):
    c=np.asarray(candidate_first,float);r=np.asarray(reference_first,float)
    if c.shape!=(36,) or r.shape!=(36,) or not np.isfinite([c,r]).all():raise ValueError('INVALID_ALIGNMENT_POSE')
    for q in (c,r):
        if abs(np.linalg.norm(q[3:7])-1)>1e-5:raise ValueError('NONUNIT_ALIGNMENT_QUATERNION')
    cr=Rotation.from_quat(c[3:7],scalar_first=True);rr=Rotation.from_quat(r[3:7],scalar_first=True)
    relative=(rr*cr.inv()).as_matrix()
    # Maximize trace(Rz(yaw)^T R_ref R_cand^T), equivalent to minimizing
    # initial SO(3) geodesic error over yaw alone.
    a=relative[0,0]+relative[1,1];b=relative[1,0]-relative[0,1]
    conditioning=float(np.hypot(a,b))
    if conditioning<1e-8:raise ValueError('INITIAL_YAW_UNOBSERVABLE')
    yaw=float(np.arctan2(b,a));turn=Rotation.from_rotvec([0,0,yaw])
    shift=r[:3]-turn.apply(c[:3]);shift[2]=0.
    return {'yaw_rad':yaw,'translation':shift.tolist(),'method':'first_frame_relative_SO3_planar_projection',
            'conditioning':conditioning,'preserve_absolute_height':True,'frames_used':[0]}
