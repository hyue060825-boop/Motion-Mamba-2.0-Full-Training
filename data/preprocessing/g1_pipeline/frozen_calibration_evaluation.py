"""Evaluate a frozen independent panel with explicit scope and review outcomes."""
import argparse
import json
import yaml
from collections import Counter,defaultdict
from .common import ROOT,read_jsonl,write_json,sha
from .calibration_protocol import require_frozen,PATH
from .gates import gate


def evaluate(namespace,panel='accept'):
    policy=require_frozen();cfg=yaml.safe_load((ROOT/policy['gate_config']).read_text())
    base=ROOT/'qa/aggregate';paths={key:base/f'{namespace}{suffix}.json' for key,suffix in
        [('run',''),('summary','_qa_summary'),('targets','_human_targets'),('relative','_relative_motion'),('joints','_joint_errors')]}
    paths['geometry']=ROOT/'work'/namespace/'pair_qa.json'
    evidence={k:json.loads(p.read_text()) for k,p in paths.items()};summary=evidence['summary'];run=evidence['run']
    if run['run_config']['policy_sha256']!=sha(PATH):raise ValueError('RUN_POLICY_MISMATCH')
    expected={'smoke':16,'dev':200,'accept':300}[panel]
    rows=read_jsonl(ROOT/f'manifests/bones_pairs_{panel}.jsonl');ids={r['sample_id'] for r in rows}
    problems=[]
    if len(rows)!=expected or len(ids)!=expected:problems.append('FIXED_INPUT_COUNT')
    for role in ('run','geometry','targets','relative','joints'):
        records=evidence[role]['results']
        if len(records)!=expected or {r['sample_id'] for r in records}!=ids:problems.append('INCOMPLETE_'+role.upper())
    if summary['candidate_contract_pass']!=expected or summary['reference_contract_pass']!=expected:problems.append('NUMERICAL_CONTRACT_FAILURE')
    if run['error_count'] or summary['measured']!=expected or evidence['targets']['measured']!=expected or evidence['joints']['measured']!=expected:
        problems.append('REQUIRED_MEASUREMENT_MISSING')
    aliases={'root_translation_rmse_m':'root_rmse_aligned_m','left_foot_position_mean_m':'left_foot_mean_m',
             'right_foot_position_mean_m':'right_foot_mean_m','left_wrist_position_mean_m':'left_wrist_mean_m',
             'right_wrist_position_mean_m':'right_wrist_mean_m','contact_f1':'contact_f1_supported_macro',
             'contact_balanced_accuracy':'contact_balanced_accuracy_supported_macro'}
    comparisons=[]
    for key,rule in cfg['paired_calibration']['metrics'].items():
        value=summary['aggregate'].get(aliases.get(key,key))
        for bound,limit in rule.items():
            metric={'clip_macro_mean_max':'clip_macro_mean','clip_macro_mean_min':'clip_macro_mean',
                    'across_clip_p95_max':'clip_p95','across_clip_p05_min':'clip_p05'}[bound]
            actual=None if value is None else value[metric]
            passed=actual is not None and (actual<=limit if bound.endswith('_max') else actual>=limit)
            comparisons.append({'metric':key,'bound':bound,'limit':limit,'actual':actual,'pass':passed})
            if not passed:problems.append('AGGREGATE_'+key+'_'+bound)
    # BA is included in the per-clip grace calculation; old development F1
    # minimum-channel rule is retained, never replaced by a more permissive macro.
    grace=0
    for row in summary['clips']:
        ba=row.get('metrics',{}).get('contact_balanced_accuracy_supported_macro')
        if row['status']=='MEASURED' and not row['measured_pairwise_grace_failures'] and (ba is None or ba>=.7):grace+=1
    if grace/expected<cfg['paired_calibration']['minimum_all_grace_pass_fraction']:problems.append('PAIRED_GRACE_FRACTION')
    for channel,stat in summary['contact_channels'].items():
        if stat['tp']+stat['fn']==0 or stat['tn']+stat['fp']==0:problems.append('CONTACT_SUPPORT_MISSING:'+channel)
    maps={k:{r['sample_id']:r for r in evidence[k]['results']} for k in ('run','geometry','targets','relative')}
    reviewpath=ROOT/'audit'/f'{namespace}_review_decisions.json';decisions={}
    if reviewpath.exists():
        review=json.loads(reviewpath.read_text())
        if review.get('policy_sha256')!=sha(PATH):raise ValueError('REVIEW_POLICY_CHANGED')
        for name,bound in review.get('bindings',{}).items():
            if sha(ROOT/name)!=bound:raise ValueError('REVIEW_EVIDENCE_CHANGED')
        # The candidate and every relevant metric report must be bound.
        required={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
        if not required.items()<=review.get('bindings',{}).items():raise ValueError('UNBOUND_REVIEW_METRICS')
        decisions={r['sample_id']:r for r in review['decisions']}
        paths['review']=reviewpath
    results=[];categories=defaultdict(lambda:{'input':0,'retained':0});pending=[]
    for sid in sorted(ids):
        r=maps['run'].get(sid,{});geom=maps['geometry'].get(sid,{});rel=maps['relative'].get(sid,{})
        errors=list(r.get('quality_rejection_reasons',[]));flags=list(r.get('review_flags',[]));core=r.get('scene_core_candidate',False)
        if r.get('status')!='DIAGNOSTIC_COMPLETE':errors.append('EXECUTION_ERROR')
        g=geom.get('candidate_geometry',{});kin=cfg['kinematic']
        if core:
            if not g or g['ground_penetration_p95_m']>kin['ground_penetration_p95_max_m'] or g['ground_penetration_max_m']>kin['ground_penetration_max_m']:errors.append('GROUND')
            if not g or g['raw_self_collision_depth_over_1cm_frame_fraction']>kin['forbidden_self_collision_frame_fraction_max'] or g['raw_self_collision_max_depth_m']>kin['forbidden_self_collision_depth_max_m']:errors.append('SELF_COLLISION')
            if r.get('candidate_continuity',{}).get('hinge_limit_excess_max_rad',float('inf'))>1e-4:errors.append('HINGE_LIMIT')
            if any(v is not None and v>kin['contact_foot_skating_p95_max_m_s'] for v in geom.get('candidate_skating_p95_m_s',{}).values()):errors.append('SKATING')
            if rel.get('status')!='MEASURED':errors.append('RELATIVE_CATEGORY_UNSUPPORTED')
            if any(v['any_coordinate_exceed_fraction']>0 for group in rel.get('roles',{}).get('candidate',{}).values() for v in group.values()):flags.append('RELATIVE_MOTION_REVIEW_REQUIRED')
        if core and flags and not errors:
            d=decisions.get(sid)
            if d is None:pending.append(sid)
            elif d.get('disposition')=='REJECT':errors.append('FROZEN_REVIEW_REJECTED')
            elif d.get('disposition')!='RETAIN_KINEMATIC_ONLY' or not d.get('rationale') or not d.get('visual_evidence'):
                raise ValueError('INVALID_REVIEW_DISPOSITION:'+sid)
            else:
                for path in d['visual_evidence']:
                    if path not in review['bindings']:raise ValueError('UNBOUND_VISUAL_REVIEW')
        retained=core and not errors and sid not in pending
        if core:
            for label in rel.get('labels',[]):categories[label]['input']+=1;categories[label]['retained']+=int(retained)
        results.append({'sample_id':sid,'core_scope':core,'scene_class':r.get('scene_class'),'kinematic_retained':retained,
                        'rejections':errors,'review_flags':flags,'review_pending':sid in pending})
    core=[r for r in results if r['core_scope']];retained=sum(r['kinematic_retained'] for r in core)
    if not core:problems.append('NO_SUPPORTED_CORE')
    elif retained/len(core)<cfg['calibration_scope']['candidate_quality_retention_within_core_min']:problems.append('CORE_QUALITY_RETENTION')
    if pending:problems.append('FROZEN_REVIEW_PENDING')
    for label,counts in categories.items():
        if counts['input']>=5 and counts['retained']==0:problems.append('CORE_CATEGORY_FAILURE:'+label)
    # A calibration gate is not a dynamic release and never accepts its
    # candidate trajectories into the official BONES training main pool.
    status='PASS' if not problems and panel=='accept' else ('DEVELOPMENT_ONLY' if not problems else 'FAIL')
    report={'status':status,'namespace':namespace,'panel':panel,'expected':expected,'attempted':len(run['results']),
        'paired_grace_pass':grace,'core_input':len(core),'core_kinematic_retained':retained,
        'unsupported_scene_count':sum(not r['core_scope'] for r in results),'review_pending_ids':pending,
        'failure_reasons':problems,'comparisons':comparisons,'category_retention':dict(categories),'results':results,
        'training_main_pool_accepted':0,'dynamic_status':'REQUIRED_AT_STAGE_4_NOT_EVALUATED_HERE',
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths.values()},'policy_sha256':sha(PATH)}
    dest=base/f'{namespace}_frozen_evaluation.json';write_json(dest,report)
    if panel=='accept':gate('01_bones_calibration','PASS' if status=='PASS' else 'FAIL',
        None if status=='PASS' else ';'.join(problems),actual_quantity=len(run['results']),
        metrics={k:report[k] for k in ('paired_grace_pass','core_input','core_kinematic_retained','unsupported_scene_count','dynamic_status')},
        input_paths=[PATH,ROOT/f'manifests/bones_pairs_{panel}.jsonl',dest,*paths.values()],
        upstream_dependencies=['00_preflight'],downstream_dependencies=['02_hml100'],
        supported_scope='source-evidenced flat ground without external support; unsupported scenes excluded from production')
    print({k:report[k] for k in ('status','paired_grace_pass','core_input','core_kinematic_retained','failure_reasons')},flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--namespace',required=True);p.add_argument('--panel',choices=['smoke','dev','accept'],default='accept');a=p.parse_args();evaluate(a.namespace,a.panel)
