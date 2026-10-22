"""HiPHI Body-only BVH -> locked project G1 development runner."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .gmr_soma import parse,HIPHI_MAP
from .locked_target import prepare,create_retargeter

# The BVH parser names are source anatomy; the target config uses the validated
# SOMA semantic names. This is an explicit table, never fuzzy name matching.
SOURCE_TO_TARGET={'Hips':'Hips','Spine2':'Chest','LeftUpLeg':'LeftLeg','LeftLeg':'LeftShin','LeftFootMod':'LeftFoot',
                  'RightUpLeg':'RightLeg','RightLeg':'RightShin','RightFootMod':'RightFoot','LeftArm':'LeftArm',
                  'LeftForeArm':'LeftForeArm','LeftHand':'LeftHand','RightArm':'RightArm','RightForeArm':'RightForeArm','RightHand':'RightHand'}


def retarget_bvh(path, *, config_path, height_m=None, max_frames=None):
    from .gates import require_pass
    require_pass('02_hml100')
    import json
    cfg=json.loads(Path(config_path).read_text())
    if cfg.get('source_adapter')!='bvh_hiphi':
        raise ValueError('HIPHI_REQUIRES_SOURCE_SPECIFIC_MAPPING_NOT_SOMA_OFFSETS')
    path=Path(path)
    # Read declared Frame Time and construct a physical 25 Hz grid without
    # integer stride or extrapolation.
    raw,_=parse(path,HIPHI_MAP,rotation_basis='world_only')
    n=len(raw);lines=path.read_text().splitlines();mi=lines.index('MOTION');dt=float(lines[mi+2].split(':',1)[1]);fps=1/dt
    if max_frames is not None:
        if not 2<=max_frames<=n: raise ValueError('INVALID_HIPHI_DIAGNOSTIC_FRAME_LIMIT')
        n=max_frames
    if n<2:raise ValueError('HIPHI_TOO_SHORT')
    target=np.arange(0,(n-1)*dt+1e-10,.04)
    frames,_=parse(path,HIPHI_MAP,target_timestamps=target,rotation_basis='world_only')
    frames=[{SOURCE_TO_TARGET[k]:v for k,v in frame.items()} for frame in frames]
    xml,_=prepare();solver=create_retargeter(config_path,xml,source_name='hiphi_native',actual_human_height=height_m)
    q=[]
    for frame in frames:q.append(solver.retarget(frame,offset_to_ground=False).copy())
    q=np.asarray(q)
    if q.shape!=(len(target),36) or not np.isfinite(q).all():raise ValueError('HIPHI_INVALID_GMR_OUTPUT')
    return q,{'frames':len(q),'source_fps':fps,'target_fps':25,'source_adapter':'hiphi_bvh_channels_local_slerp_fk','height_m':height_m,'finite':True}
