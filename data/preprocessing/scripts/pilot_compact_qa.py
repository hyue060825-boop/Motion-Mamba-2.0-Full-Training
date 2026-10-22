"""Read existing evidence into one pilot-only compact three-source report."""
import json
from collections import Counter
import numpy as np
import yaml
from g1_pipeline.common import ROOT, read_jsonl, sha, write_json, now
from g1_pipeline.pilot_scope import require_pilot


def load(name):
    return json.loads((ROOT / name).read_text())


def stats(rows, key):
    v = [float(np.max(r[key])) if isinstance(r.get(key),list) else r[key]
         for r in rows if isinstance(r.get(key),(int,float)) or isinstance(r.get(key),list) and len(r[key])]
    return {'measured_clips':len(v), 'mean':float(np.mean(v)) if v else None,
            'p95':float(np.percentile(v,95)) if v else None, 'max':max(v) if v else None,
            'vector_reduction':'maximum over joints within each clip' if any(isinstance(r.get(key),list) for r in rows) else None}


def source_summary(short):
    ns = load(f'audit/current_pilot_{short}100_run.json')['namespace']
    records = read_jsonl(ROOT / 'work' / ns / 'results.jsonl')
    a = load(f'qa/aggregate/{ns}_acceptance.json')
    assert a['execution_scope'] == 'isolated_pilot' and len(records) == 100
    contracts=[]; geometry=[]; continuity=[]; targets=[]; core=0; core_kept=0
    retained_contracts=[]; retained_frames=0
    outcomes={r['sample_id']:r for r in a['results']}
    for r in records:
        is_core=r.get('source_semantics',{}).get('scene_class')=='flat_ground_core_candidate'
        core+=int(is_core); core_kept+=int(is_core and outcomes[r['sample_id']]['quality_pass'])
        if not r.get('output_paths'):continue
        for p,h in r['output_hashes'].items():
            if sha(ROOT/p)!=h:raise ValueError('PILOT_OUTPUT_CHANGED:'+p)
        c=load(r['output_paths']['contract_path']);m=load(r['metrics_path'])
        contracts.append(c);geometry.append(m['geometry']);continuity.append(m['continuity'])
        targets.append(m['human_target_metrics'])
        if outcomes[r['sample_id']]['quality_pass']:
            retained_contracts.append(c);retained_frames+=r['frames']
    errors=sum(r['status']=='ERROR' for r in records)
    pending=len(a['review_pending_ids']);accepted=a['kinematic_retained']
    result={'namespace':ns,'scope':'isolated pilot regression/quality panel, not independent held-out',
        'expected':100,'attempted':len(records),'accepted':accepted,
        'rejected':100-accepted-errors-pending,'error':errors,'review_pending':pending,
        'pilot_status':a['status'],'failure_reasons':a['failure_reasons'],
        'frames':sum(r.get('frames',0) for r in records),
        'accepted_frames':retained_frames,'accepted_frame_coverage_s':retained_frames/25,
        'duration_last_sample_sum_s':sum(r.get('last_sample_time_sec',0) for r in records),
        'duration_frame_coverage_s':sum(r.get('frames',0)/25 for r in records),
        'flat_core_input':core,'flat_core_retained':core_kept,
        'flat_core_retention':core_kept/core if core else None,
        'contract_pass':sum(c['status']=='PASS' for c in contracts),
        'contract_measured':len(contracts),
        'accepted_contract_pass':sum(c['status']=='PASS' for c in retained_contracts),
        'all_exported_contract_shape_time_finite':all(c['finite'] and c['fps']==25 and c['feature_shape']==[c['frames'],414] and c['project_loader_length']==c['frames'] for c in contracts),
        'accepted_skating_p95_max_m_s':max([v for c in retained_contracts for v in c['skating_p95_m_s'].values() if v is not None],default=None),
        'hard_rejection_counts':dict(Counter(x for r in a['results'] for x in r['rejections'])),
        'review_flag_counts':dict(Counter(x for r in records for x in r.get('review_flags',[]))),
        'groups':a['groups'],
        'geometry':{k:stats(geometry,k) for k in ['ground_penetration_p95_m','ground_penetration_max_m',
            'raw_self_collision_depth_over_1cm_frame_fraction','raw_self_collision_max_depth_m']},
        'continuity':{k:stats(continuity,k) for k in ['hinge_limit_excess_max_rad','root_translation_speed_p95_m_s',
            'root_angular_speed_p95_rad_s','root_linear_acceleration_max_m_s2','hinge_speed_max_rad_s',
            'hinge_acceleration_max_rad_s2','timestep_max_error_s','unit_quaternion_max_error']},
        'contract_errors':{k:stats(contracts,k) for k in ['fk_position_max_m','fk_orientation_max_deg',
            'inverse_position_max_m','project_loader_max_abs','contact_mismatches']},
        'skating_p95_max_m_s':max([v for c in contracts for v in c['skating_p95_m_s'].values() if v is not None], default=None),
        'target_body_clip_mean_position_m':{b:stats([x[b] for x in targets], 'position_mean_m') for b in targets[0]} if targets else {},
        'review_method':'Agent sampled-frame inspection plus full-series numeric QA; no human signature, exhaustive visual review or dynamic acceptance claimed',
        'systematic_checks':load(f'audit/{ns}_review_decisions.json')['systematic_checks'],
        'systematic_check_rationale':load(f'audit/{ns}_review_decisions.json')['systematic_check_rationale'],
        'all_clip_reviews_completed':len(load(f'audit/{ns}_review_decisions.json')['decisions'])==100 and pending==0,
        'evidence':{f'qa/aggregate/{ns}_acceptance.json':sha(ROOT/f'qa/aggregate/{ns}_acceptance.json'),
                    f'work/{ns}/results.jsonl':sha(ROOT/f'work/{ns}/results.jsonl')},
        'production_accepted':0}
    assert result['accepted']+result['rejected']+result['error']+result['review_pending']==100
    return result


def bones_summary():
    ns='bones_frozen_accept_v25'
    geometry=load(f'work/{ns}/pair_qa.json')['results']
    run={r['sample_id']:r for r in load(f'qa/aggregate/{ns}.json')['results']}
    k=yaml.safe_load((ROOT/'configs/gates.v2.yaml').read_text())['kinematic']
    core=0; passed=0; rows=[];groups={}
    for r in geometry:
        g=r['reference_geometry'];c=run[r['sample_id']]['reference_continuity']
        reasons=[]
        if r['reference_contract']['status']!='PASS':reasons.append('CONTRACT')
        if c['hinge_limit_excess_max_rad']>1e-4:reasons.append('JOINT_RANGE')
        is_core=run[r['sample_id']]['scene_core_candidate']
        if is_core:
            core+=1
            if g['ground_penetration_p95_m']>k['ground_penetration_p95_max_m'] or g['ground_penetration_max_m']>k['ground_penetration_max_m']:reasons.append('GROUND')
            if g['raw_self_collision_depth_over_1cm_frame_fraction']>k['forbidden_self_collision_frame_fraction_max'] or g['raw_self_collision_max_depth_m']>k['forbidden_self_collision_depth_max_m']:reasons.append('SELF_COLLISION')
            if any(v is not None and v>k['contact_foot_skating_p95_max_m_s'] for v in r['reference_skating_p95_m_s'].values()):reasons.append('SKATING')
            passed+=int(not reasons)
        rows.append({'sample_id':r['sample_id'],'core_scope':is_core,'diagnostic_hard_failures':reasons})
        group=groups.setdefault(r['source_category'],{'input':0,'core_input':0,'core_hard_qa_pass':0})
        group['input']+=1;group['core_input']+=int(is_core);group['core_hard_qa_pass']+=int(is_core and not reasons)
    return {'scope':'READ_ONLY official reference side of existing accept300 paired QA; cropped paired time support, held-out comparison only; not a training pool',
        'expected':300,'attempted':len(geometry),'measured':sum(r['status']=='MEASURED' for r in geometry),
        'production_accepted':0,'accepted':None,'accepted_reason':'No official production acceptance attempted; these are existing reference diagnostics',
        'rejected':None,'error':sum(r['status']!='MEASURED' for r in geometry),'groups':groups,
        'frames':sum(r['frames'] for r in geometry),'duration_last_sample_sum_s':sum((r['frames']-1)/25 for r in geometry),
        'contract_pass':sum(r['reference_contract']['status']=='PASS' for r in geometry),
        'contract_errors':{k:stats([r['reference_contract'] for r in geometry],k) for k in ['inverse_hinge_max_deg','inverse_rotation_max_deg','inverse_translation_max_m','encoder_contact_mismatches']},
        'skating_p95_max_m_s':max([v for r in geometry for v in r['reference_skating_p95_m_s'].values() if v is not None],default=None),
        'diagnostic_hard_failure_counts':dict(Counter(x for r in rows for x in r['diagnostic_hard_failures'])),
        'flat_core_input':core,'flat_core_hard_qa_pass':passed,'flat_core_hard_qa_fraction':passed/core,
        'geometry':{k:stats([r['reference_geometry'] for r in geometry],k) for k in ['ground_penetration_p95_m','ground_penetration_max_m','raw_self_collision_depth_over_1cm_frame_fraction','raw_self_collision_max_depth_m']},
        'continuity':{k:stats([r['reference_continuity'] for r in run.values()],k) for k in ['hinge_limit_excess_max_rad','root_translation_speed_p95_m_s','root_angular_speed_p95_rad_s','root_linear_acceleration_max_m_s2','hinge_speed_max_rad_s','hinge_acceleration_max_rad_s2']},
        'results':rows,'original_candidate_acceptance':'FAIL 112/118, unchanged',
        'evidence':{f'work/{ns}/pair_qa.json':sha(ROOT/f'work/{ns}/pair_qa.json')}}


def frozen_candidate_summary():
    path='qa/aggregate/bones_frozen_accept_v25_frozen_evaluation.json'
    frozen=load(path)
    assert frozen['status']=='FAIL' and frozen['core_input']==118 and frozen['core_kinematic_retained']==112
    return {k:frozen[k] for k in ['status','attempted','core_input','core_kinematic_retained',
        'paired_grace_pass','failure_reasons','policy_sha256','training_main_pool_accepted']} | {
        'scope':'Original frozen candidate acceptance, read-only; distinct from official-reference diagnostics',
        'evidence':{path:sha(ROOT/path)},'permanent_result_preserved':True,
        'unsupported_readonly_report':'reports/pilot_bones_unsupported_readonly.md'}


if __name__=='__main__':
    require_pilot('humanml3d');require_pilot('hiphi')
    report={'time':now(),'status':'PILOT_COMPARISON_ONLY','training_data_ready':False,
        'bones_frozen_candidate_acceptance':frozen_candidate_summary(),
        'sources':{'bones_official_reference':bones_summary(),'humanml3d':source_summary('hml'),'hiphi':source_summary('hiphi')},
        'common_contracts':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'contracts').glob('*.json')},
        'shared_semantics':{'qpos_width':36,'physical_dof':29,'skeleton_nodes':34,'feature_width':414,'fps':25,
            'normalizer':'unchanged G1 contract; no new stats','source_basis':'source-specific native adapters followed by locked common target/exporter/loader'},
        'production_missing':['Original BONES calibration remains permanently FAIL; pilot authorization grants no production admission',
            'Source quality and systematic review failures, if present, must remain recorded',
            'No authorized and passed production HML/HiPHI gates',
            'No three-source dynamic sanity acceptance',
            'No canary/full batch, final split/leakage audit, train-only statistics or full mixed training-loader and model smoke acceptance'],
        'scope_limit':'No canary, full batch, formal training or TRAINING_DATA_READY. Different category distributions are reported, not forced equal.'}
    write_json(ROOT/'qa/aggregate/pilot_three_source_compact_qa.json',report)
    print({s:{k:r.get(k) for k in ['attempted','accepted','rejected','error','review_pending','pilot_status','flat_core_hard_qa_pass']} for s,r in report['sources'].items()})
