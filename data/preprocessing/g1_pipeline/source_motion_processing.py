"""Shared gated human-frame -> GMR -> corrected G1 production candidate path."""
import json
import numpy as np
from .common import ROOT
from .gates import require_pass
from .calibration_protocol import require_frozen
from .locked_target import prepare,create_retargeter
from .gmr_temporal_arms import solve
from .gmr_collision_refinement import refine
from .ground_clearance import correct
from .bounded_hinge_trajectory import project,sonic_training_reference_limits
from .bones_pair_qa import replay
from .bones_continuity_qa import metrics as continuity
from .bones_relative_motion_v2 import series
from .frozen_calibration_runner import target_errors
from .source_posture import support_evidence


def process(frames,timestamps,config_path,semantics,*,dataset,pilot=False):
    prerequisite={'humanml3d':'01_bones_calibration','hiphi':'02_hml100'}[dataset]
    if pilot:
        from .pilot_scope import require_pilot
        require_pilot(dataset)
    else:require_pass(prerequisite)
    policy=require_frozen();cfg=json.loads(config_path.read_text())
    if cfg.get('source_adapter') not in ('native_fk_to_gmr_smplx_semantics','hiphi_native_fk_source_specific_rest'):
        raise ValueError('UNVALIDATED_SOURCE_MAPPING')
    if len(frames)!=len(timestamps) or len(frames)<3:raise ValueError('INVALID_SOURCE_TIME_SUPPORT')
    xml,_=prepare();solver=create_retargeter(config_path,xml,source_name=dataset+'_native_frames')
    q,temporal=solve(solver,frames,cfg['temporal_arms'],cfg['weighted_convergence'])
    initial_qpos=q.copy();corrected=[];collisions=[];delta=None
    for frame,seed in zip(frames,q):
        pose,info=refine(solver,frame,seed,previous_correction=delta,**policy['collision_settings'])
        delta=pose[7:]-seed[7:];corrected.append(pose);collisions.append(info)
    original=np.asarray(corrected);q=original.copy();post=json.loads((ROOT/policy['post_config']).read_text())
    rejected=[];flags=[];scene=semantics['scene_class'];ground=None;rate=None
    m=solver.model;before_geom=replay(m,original)[2]
    if scene in ('flat_ground_core_candidate','ground_multisupport'):
        if before_geom['ground_penetration_p95_m']>post['ground_trigger_p95_m'] or before_geom['ground_penetration_max_m']>post['ground_trigger_max_m']:
            q,ground=correct(m,q,permitted_penetration=post['ground_target_penetration_m'])
        q[:,7:],rate=project(q[:,7:],m.jnt_range[1:,0],m.jnt_range[1:,1],sonic_training_reference_limits(m),
            dt=.04,correction_curvature=post['rate_projection_correction_curvature'])
        if rate['correction_max_rad']>post['maximum_rate_projection_angle_change_rad']:rejected.append('RATE_CORRECTION_EXCEEDS_CONTENT_CHANGE_BUDGET')
    else:rejected.append('EXTERNAL_OR_UNRESOLVED_SUPPORT_SCENE')
    pe0,oe0=target_errors(solver,frames,original,cfg);pe,oe=target_errors(solver,frames,q,cfg)
    changes={'position_worst_increase_m':float((pe-pe0).max()),'orientation_worst_increase_deg':float((oe-oe0).max())}
    guard=policy['human_target_correction_review']
    if changes['position_worst_increase_m']>guard['position_increase_above_m'] or changes['orientation_worst_increase_deg']>guard['orientation_increase_above_deg']:
        flags.append('CORRECTION_CONTENT_REVIEW_REQUIRED')
    geometry=replay(m,q)[2];cont=continuity(m,q,timestamps)
    baseline=json.loads((ROOT/policy['relative_baseline']).read_text())['baseline'];values=series(m,q,timestamps)
    relative={}
    for label in semantics['action_labels']:
        if label not in baseline or not baseline[label]['supported']:continue
        relative[label]={}
        for key,a in values.items():
            threshold=np.array(baseline[label]['thresholds'][key]);exceed=a>threshold
            relative[label][key]={'any_coordinate_exceed_fraction':float(exceed.any(axis=1).mean()),
                'max_threshold_ratio':float(np.max(a/threshold)),'flagged_sample_indices':np.flatnonzero(exceed.any(axis=1)).tolist(),
                'per_coordinate_max':a.max(axis=0).tolist()}
    if not relative:rejected.append('RELATIVE_ACTION_CLASS_UNSUPPORTED')
    elif any(v['any_coordinate_exceed_fraction']>0 for group in relative.values() for v in group.values()):flags.append('RELATIVE_MOTION_REVIEW_REQUIRED')
    # Native posture remains a review signal; a deep crouch is not silently
    # converted into a different scene because its robot QA is inconvenient.
    lookup={b:cfg['ik_match_table2'][b][0] for b in cfg['ik_match_table2']}
    def points(bodies):return np.array([[f[lookup[b]][0] for b in bodies] for f in frames])
    posture=support_evidence(points(['pelvis'])[:,0],points(['torso_link'])[:,0],
        points(['left_knee_link','right_knee_link']),points(['left_ankle_roll_link','right_ankle_roll_link']),cfg['scale_strategy']['source_height_m'])
    if posture['ground_multisupport_review'] and scene=='flat_ground_core_candidate':flags.append('NATIVE_SUPPORT_POSTURE_REVIEW_REQUIRED')
    if geometry['ground_penetration_p95_m']>.02 or geometry['ground_penetration_max_m']>.05:rejected.append('GROUND_PENETRATION')
    if geometry['raw_self_collision_depth_over_1cm_frame_fraction']>.001 or geometry['raw_self_collision_max_depth_m']>.03:rejected.append('SELF_COLLISION')
    if cont['hinge_limit_excess_max_rad']>1e-4:rejected.append('HINGE_LIMIT')
    details={'temporal':temporal,'collision_frames':collisions,'ground_correction':ground,'rate_projection':rate,
        'source_semantics':semantics,'human_posture':posture,'geometry':geometry,'original_geometry':before_geom,
        'continuity':cont,'relative_motion':relative,'human_target_residual_change':changes,
        'human_target_body_order':list(cfg['ik_match_table2']),
        'human_target_metrics':{b:{'position_mean_m':float(pe[:,j].mean()),'position_p95_m':float(np.percentile(pe[:,j],95)),
            'position_max_m':float(pe[:,j].max()),'orientation_mean_deg':float(oe[:,j].mean()),
            'orientation_p95_deg':float(np.percentile(oe[:,j],95)),'orientation_max_deg':float(oe[:,j].max())} for j,b in enumerate(cfg['ik_match_table2'])},
        'rejections':rejected,'review_flags':flags,'hardware_velocity_compliance':None,
        'hardware_velocity_reason':'training-config rate regularization and relative flags do not establish motor capability'}
    target_positions=[]
    for frame in frames:
        solver.update_targets(frame,offset_to_ground=False)
        target_positions.append([solver.scaled_human_data[cfg['ik_match_table2'][b][0]][0].copy() for b in cfg['ik_match_table2']])
    return q,original,details,{'position_error_m':pe.astype(np.float32),'orientation_error_deg':oe.astype(np.float32),
        'scaled_target_positions_m':np.asarray(target_positions,dtype=np.float32),'initial_gmr_qpos':initial_qpos}
