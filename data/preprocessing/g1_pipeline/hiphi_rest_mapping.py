"""HiPHI per-file rest mapping from exported rig, with explicit actor-height audit."""
import numpy as np
from .common import sha
from .hiphi_native import read_bvh,world_fk,WORLD_BASIS,LANDMARKS
from .native_rest_mapping import build_mapping

HIPHI_MAP={'pelvis':'Hips','torso_link':'Spine4'}
for side,hside in [('left','Left'),('right','Right')]:
    for robot,human in [('hip_yaw','UpLeg'),('knee','Leg'),('ankle_roll','Foot'),
                        ('shoulder_yaw','Arm'),('elbow','ForeArm'),('wrist_yaw','Hand')]:
        HIPHI_MAP[f'{side}_{robot}_link']=hside+human


def mapping(path,metadata_height):
    src=read_bvh(path);nodes=src['nodes'];ids={n['name']:i for i,n in enumerate(nodes)}
    if len([n for n in nodes if not n['end_site']])!=55 or sum(n['end_site'] for n in nodes)!=13 or not set(LANDMARKS)<=set(ids):
        raise ValueError('HIPHI_REQUIRED_SOURCE_ANATOMY_MISSING')
    # Position channels are actual local transforms in this BVH dialect. Derive
    # a zero-rotation rest from their first-frame values, then preserve all
    # subsequent translations in the motion FK. Root translation is removed
    # only for static rig measurement, never from the source trajectory.
    local=src['local_positions_m'][0].copy();local[0]=0
    rest,_=world_fk(local[None],np.tile(np.eye(3),(1,len(nodes),1,1)),[n['parent'] for n in nodes])
    p=rest[0];floor=float(p[:,1].min());height=float(np.ptp(p[:,1]));pelvis=float(p[0,1]-floor)
    canonical=(p-p[0])@WORLD_BASIS.T+np.array([0,0,pelvis])
    positions={n:canonical[ids[n]] for n in LANDMARKS};rotations={n:WORLD_BASIS for n in LANDMARKS}
    metadata_height=float(metadata_height)
    if not .8<metadata_height<2.5:raise ValueError('INVALID_ACTOR_HEIGHT_METADATA')
    ratios=height/metadata_height
    local_length=np.linalg.norm(src['local_positions_m'][:,1:],axis=-1)
    cfg,audit=build_mapping(positions,rotations,HIPHI_MAP,pelvis,height,
        source_adapter='hiphi_native_fk_source_specific_rest',root_name='Hips',
        provenance={'native_path':str(path),'native_sha256':sha(path),'metadata_actor_height_m':metadata_height,
                    'exported_rig_height_proxy_m':height,'rig_proxy_over_actor_height':ratios,
                    'height_discrepancy_above_10_percent':abs(ratios-1)>.1,
                    'height_disposition':'use actual exported rig dimensions; actor height is provenance and discrepancy evidence, not a second scale',
                    'height_proxy_definition':'zero-rotation first-frame local translations including End Sites; not measured actor body height',
                    'nonroot_segment_length_variation_max_m':float(np.ptp(local_length,axis=0).max()),
                    'native_dynamic_translations_preserved':True,'source_fps':src['fps'],'source_frames':src['frames'],
                    'rotation_basis':'world-only WORLD_BASIS; source joint axes retained',
                    'rest_floor_definition':'minimum Y over zero-rotation exported hierarchy including End Sites'})
    cfg['scale_strategy'].update(metadata_actor_height_m=metadata_height,source_height_is_skeleton_proxy=True,
        exported_rig_over_actor_height=ratios)
    return cfg,audit
