"""Native SMPL-H shape/DMPL geometry; robot retargeting requires upstream gates."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from .common import HML
from .hml_native import native_pose_to_frames


@lru_cache(maxsize=3)
def model_geometry(base, gender):
    base=Path(base)
    if gender not in ('male','female','neutral'):raise ValueError('UNSUPPORTED_NATIVE_GENDER')
    model_path=base/f'body_models/smplh/{gender}/model.npz'
    dmpl_path=base/f'body_models/dmpls/{gender}/model.npz'
    with np.load(model_path,allow_pickle=False) as model, np.load(dmpl_path,allow_pickle=False) as dmpl:
        tree=model['kintree_table'];parents=tree[0].astype(np.int64);parents[0]=-1
        if len(parents)!=52 or not np.array_equal(tree[1],np.arange(52)):raise ValueError('NOT_SMPLH52_NATIVE_MODEL')
        reg=np.asarray(model['J_regressor'],float);template=np.asarray(model['v_template'],float)
        shape=np.asarray(model['shapedirs'][...,:16],float);dynamic=np.asarray(dmpl['eigvec'][...,:8],float)
        # Linearity of joint regression avoids native-rate vertex skinning.
        return {'parents':parents,'template':template,'shape':shape,'rest':reg@template,
                'joint_shape':np.einsum('jv,vck->jck',reg,shape),
                'joint_dynamic':np.einsum('jv,vck->jck',reg,dynamic),
                'model_path':str(model_path),'dmpl_path':str(dmpl_path)}


def native_parameters(npz_path, *, start_frame, end_frame, body_model_root=None):
    """Read supported SMPL-H parameters with explicit bounds and field checks."""
    base=Path(body_model_root) if body_model_root else HML
    with np.load(npz_path,allow_pickle=False) as z:
        poses=np.asarray(z['poses'],float);total=len(poses)
        if poses.ndim!=2 or poses.shape[1]!=156:raise ValueError('HML_NATIVE_REQUIRES_SMPLH_156_POSE')
        if not isinstance(start_frame,(int,np.integer)) or not isinstance(end_frame,(int,np.integer)) or not 0<=start_frame<end_frame<=total:
            raise ValueError(f'NATIVE_INTERVAL_EXCEEDS_SOURCE_FRAMES:{start_frame}:{end_frame}:{total}')
        fps_values=[float(z[k]) for k in ('mocap_framerate','mocap_frame_rate') if k in z]
        if not fps_values or len(set(fps_values))!=1 or not np.isfinite(fps_values[0]) or fps_values[0]<=0:raise ValueError('FPS_MISSING_OR_CONFLICT')
        gender=z['gender'].item();gender=gender.decode() if isinstance(gender,bytes) else str(gender)
        betas=np.asarray(z['betas'],float);trans=np.asarray(z['trans'],float);dmpls=np.asarray(z['dmpls'],float)
        if betas.shape!=(16,) or trans.shape!=(total,3) or dmpls.shape!=(total,8):raise ValueError('UNSUPPORTED_NATIVE_PARAMETER_LAYOUT')
        if not all(np.isfinite(a).all() for a in (poses,trans,dmpls,betas)):raise ValueError('NONFINITE_NATIVE_PARAMETER')
        if 'body_model_type' in z and str(z['body_model_type'].item()).lower() not in ('smplh','smpl-h'):raise ValueError('CONFLICTING_NATIVE_MODEL_TYPE')
    geom=model_geometry(str(base),gender);sel=slice(start_frame,end_frame)
    rest=geom['rest'][None]+np.einsum('jck,k->jc',geom['joint_shape'],betas)[None]+np.einsum('jck,nk->njc',geom['joint_dynamic'],dmpls[sel])
    offsets=rest.copy();offsets[:,1:]-=rest[:,geom['parents'][1:]]
    return {'full_pose':poses[sel].reshape(-1,52,3),'trans':trans[sel],'parents':geom['parents'],
            'offsets':offsets,'source_fps':fps_values[0],'start_frame':start_frame,
            'gender':gender,'betas':betas,'dmpls':dmpls[sel],'geometry':geom}


def load_native_frames(npz_path, *, start_frame, end_frame, target_timestamps, body_model_root=None):
    p=native_parameters(npz_path,start_frame=start_frame,end_frame=end_frame,body_model_root=body_model_root)
    return native_pose_to_frames(p['full_pose'],p['trans'],p['parents'],p['offsets'],p['source_fps'],
                                 target_timestamps,start_frame=start_frame)


def estimate_rest_height(npz_path, *, body_model_root=None):
    """Canonical mesh head-to-sole extent, not the shorter joint-center span."""
    p=native_parameters(npz_path,start_frame=0,end_frame=1,body_model_root=body_model_root)
    g=p['geometry'];vertices=g['template']+np.einsum('vck,k->vc',g['shape'],p['betas'])
    height=float(np.ptp(vertices[:,1]))
    if not np.isfinite(height) or not .5<height<3.:raise ValueError('INVALID_NATIVE_REST_HEIGHT')
    return height


def retarget_clip(npz_path, *, start_frame, end_frame, target_timestamps, config_path, xml_path=None, actual_human_height=None):
    from .gates import require_pass
    require_pass('01_bones_calibration')
    cfg=json.loads(Path(config_path).read_text())
    if cfg.get('source_adapter')!='native_fk_to_gmr_smplx_semantics':
        raise ValueError('HML_REQUIRES_SOURCE_SPECIFIC_MAPPING_NOT_SOMA_OFFSETS')
    # The native geometry below is independently testable. Source-specific
    # target rest offsets and scale still require their own frozen audit.
    raise RuntimeError('HML_ROBOT_MAPPING_NOT_YET_VALIDATED')
