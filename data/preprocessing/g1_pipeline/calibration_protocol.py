"""Freeze development-derived policy once, before reading held-out trajectories."""
import json
import yaml
from .common import ROOT,read_jsonl,write_json,sha,atomic_bytes,now
from pathlib import Path
from .gates import require_pass

PATH=ROOT/'configs/calibration_protocol.v25.json'


def require_frozen():
    require_pass('00_preflight')
    if not PATH.is_file():raise RuntimeError('CALIBRATION_POLICY_NOT_FROZEN')
    policy=json.loads(PATH.read_text())
    if policy['status']!='FROZEN_BEFORE_HELD_OUT':raise RuntimeError('INVALID_CALIBRATION_FREEZE')
    for name,expected in policy['bindings'].items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=expected:raise RuntimeError('FROZEN_CALIBRATION_INPUT_CHANGED:'+name)
    return policy


def freeze():
    require_pass('00_preflight')
    if PATH.exists():return require_frozen()
    review=json.loads((ROOT/'audit/bones_core_v24_review.json').read_text())
    video=json.loads((ROOT/'videos/bones_core_dev_v24/manifest.json').read_text())
    summary=json.loads((ROOT/'qa/aggregate/bones_core_dev_v24_qa_summary.json').read_text())
    if review['measured']!=200 or review['core_ground_pass']!=review['core_clips'] or review['core_self_collision_pass']!=review['core_clips']:
        raise ValueError('DEVELOPMENT_PHYSICS_INCOMPLETE')
    if summary['candidate_contract_pass']!=200 or summary['reference_contract_pass']!=200 or summary['pairwise_grace_pass_count']<190:
        raise ValueError('DEVELOPMENT_PAIRED_REQUIREMENTS_FAILED')
    if video['rendered']!=20:raise ValueError('DEVELOPMENT_VISUAL_EVIDENCE_INCOMPLETE')
    for row in video['videos']:
        if sha(row['path'])!=row['sha256']:raise ValueError('VIDEO_CHANGED')
    # Only metadata is read for leakage checks. No acceptance motion is opened.
    dev=read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl');accept=read_jsonl(ROOT/'manifests/bones_pairs_accept.jsonl')
    if len(accept)!=300 or len({r['source_family_id'] for r in accept})!=300:raise ValueError('ACCEPT_300_INDEPENDENT_REQUIRED')
    if {r['source_family_id'] for r in dev}&{r['source_family_id'] for r in accept}:raise ValueError('DEV_ACCEPT_FAMILY_LEAKAGE')
    if any(r['project_split']!='train' or r['adapter_split']!='accept' for r in accept):raise ValueError('HELD_OUT_SPLIT_MISMATCH')
    cfg=yaml.safe_load((ROOT/'configs/gates.v1.yaml').read_text())
    cfg.update(policy_version='gates.v2-frozen-development',status='FROZEN_BEFORE_HELD_OUT',release_allowed=False)
    cfg['freeze_requires']=['fixed_dev200_complete','source_scene_and_collision_policy','frozen_relative_baseline','bound_code_config_assets_and_panels']
    cfg['kinematic']['allowed_collision_policy']={'extra_allowed_pairs':[], 'base_filter':'locked compiled MuJoCo collision masks, explicit XML excludes and parent filtering; no task-specific exemption'}
    cfg['kinematic']['allowed_collision_policy_reason']=None
    cfg['kinematic']['relative_outlier_numerical_floor']=json.loads((ROOT/'qa/aggregate/bones_core_dev_v24_relative_motion_v2.json').read_text())['numerical_iqr_floors']
    cfg['kinematic']['relative_outlier_floor_reason']=None
    cfg['calibration_scope']={'pairwise_denominator':300,'geometry_scope':'source_evidenced_flat_ground_core',
        'unsupported_scenes':'retain all paired measurements, exclude from training eligibility until separate scene validation',
        'candidate_quality_retention_within_core_min':.95,'all_training_eligible_numeric_and_geometry_pass':True,
        'absolute_hardware_velocity_required_at_calibration':False,
        'reason':'task section 7.4 permits relative QA when matched specifications are absent; full dynamic comparison remains stage 4 mandatory',
        'relative_flags':'REVIEW_REQUIRED, never automatic hardware failure or automatic acceptance',
        'contact_grace':'minimum supported-channel F1 >=0.70; supported macro balanced accuracy >=0.70; unknown support stays null',
        'support_requirement':'each contact channel must have positive/negative reference support in aggregate',
        'category_failure':'any supported core action class with zero quality-retained clips blocks; rare classes reported without inflated confidence'}
    v2=ROOT/'configs/gates.v2.yaml';atomic_bytes(v2,yaml.safe_dump(cfg,sort_keys=False).encode())
    files=[ROOT/'configs/gates.v2.yaml',ROOT/'configs/bones_temporal_arms_v15.json',ROOT/'configs/bones_core_refinement_v24.json',
        ROOT/'qa/aggregate/bones_core_dev_v24_relative_motion_v2.json',ROOT/'audit/bones_core_v24_review.json',
        ROOT/'reports/bones_core_v24_development_review.md',ROOT/'videos/bones_core_dev_v24/manifest.json',
        ROOT/'contracts/g1_robot.json',ROOT/'contracts/bones_g1_csv.json',ROOT/'contracts/g1_skeleton34.json',ROOT/'contracts/motion_mamba2_414.json']
    files += list((ROOT/'g1_pipeline').glob('*.py'))
    files += list((ROOT/'deps/GMR/general_motion_retargeting').rglob('*.py'))
    # Bind the actually audited external loader/encoder/project sources too.
    files += [Path(p) for p in json.loads((ROOT/'audit/project_source_hashes.json').read_text())]
    files += [ROOT/'audit/versions.lock.json']
    files += [ROOT/f'manifests/bones_pairs_{p}.jsonl' for p in ('smoke','dev','accept')]
    policy={'status':'FROZEN_BEFORE_HELD_OUT','time':now(),'semantic_version':'g1-bones-calibration-v25',
        'solver_config':'configs/bones_temporal_arms_v15.json','post_config':'configs/bones_core_refinement_v24.json',
        'gate_config':'configs/gates.v2.yaml','relative_baseline':'qa/aggregate/bones_core_dev_v24_relative_motion_v2.json',
        'collision_settings':{'penalty':2000.,'posture_weight':5.,'max_nfev':80,'correction_velocity_weight':.4},
        'held_out_expected':300,'held_out_trajectories_read_at_freeze':False,'reference_in_solver_objective':False,
        'human_target_correction_review':{'position_increase_above_m':.03,'orientation_increase_above_deg':5.,
          'meaning':'new correction-induced content change triggers review/rejection; baseline residuals remain separately reported'},
        'resources':{'cpu_workers':1,'cpu_threads':1,'gpu':False,'estimated_hours':4.,'estimated_disk_gb':4.,'stop_free_disk_below_gb':20},
        'bindings':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in files}}
    write_json(PATH,policy);return policy


if __name__=='__main__':print(freeze()['status'])
