"""Fixed-denominator source acceptance with bound numerical and visual evidence."""
import argparse
import json
from collections import Counter,defaultdict
import numpy as np
import yaml
from .common import ROOT,HML,read_jsonl,write_json,write_jsonl,sha,digest
from .gates import require_pass,gate as production_gate
from .pilot_scope import require_pilot,pilot_gate,AUTH
from .calibration_protocol import require_frozen,PATH


def select_videos(namespace):
    rows=read_jsonl(ROOT/'work'/namespace/'results.jsonl');usable=[r for r in rows if r.get('output_paths')]
    def score(row):
        m=json.loads((ROOT/row['metrics_path']).read_text())
        return max([v['position_max_m'] for v in m['human_target_metrics'].values()]+[0])
    ranked=sorted(usable,key=score,reverse=True);selected={r['sample_id']:'worst_human_target_residual' for r in ranked[:8]}
    groups=defaultdict(list)
    for row in usable:
        if row['sample_id'] not in selected:groups[row['source_semantics']['scene_class']].append(row)
    for group in groups.values():group.sort(key=lambda r:digest([20260915,r['sample_id']]))
    while len(selected)<min(20,len(usable)) and any(groups.values()):
        for key,group in sorted(groups.items()):
            if group and len(selected)<20:selected[group.pop(0)['sample_id']]='source_scene_stratified_'+key
    for row in usable:
        if row.get('review_flags') and not row.get('rejections'):selected.setdefault(row['sample_id'],'required_anomaly_review')
    return selected


def evaluate(namespace,source,*,pilot=False):
    spec={'humanml3d':('02_hml100','01_bones_calibration','hml_smoke100.v2.jsonl','production_hml'),
          'hiphi':('03_hiphi100','02_hml100','hiphi_smoke100.jsonl','production_hiphi')}
    stage,prior,filename,module=spec[source]
    if pilot:require_pilot(source)
    else:require_pass(prior)
    if namespace.startswith('pilot_') != pilot:raise ValueError('SOURCE_EXECUTION_SCOPE_MISMATCH')
    gate=pilot_gate if pilot else production_gate
    policy=require_frozen()
    base=ROOT/'work'/namespace;manifest=ROOT/'manifests'/filename;rows=read_jsonl(manifest);ledger=base/'results.jsonl';records=read_jsonl(ledger)
    design=json.loads((base/'run.json').read_text());issues=[]
    if design.get('execution_scope','production') != ('isolated_pilot' if pilot else 'production'):raise ValueError('SOURCE_EXECUTION_SCOPE_MISMATCH')
    if pilot and (design['fingerprint']['pilot_authorization']!=sha(AUTH) or design['fingerprint']['pilot_scope_code']!=sha(ROOT/'g1_pipeline/pilot_scope.py')):raise ValueError('PILOT_EXECUTION_FINGERPRINT_CHANGED')
    if design['fingerprint']['manifest']!=sha(manifest):raise ValueError('SOURCE_MANIFEST_CHANGED')
    staticpath=ROOT/'audit'/('hml_native_rest_mapping_v2.json' if source=='humanml3d' else 'hiphi_native_rest_mapping_v1.json')
    if sha(staticpath)!=design['fingerprint']['static_mappings']:raise ValueError('SOURCE_STATIC_EVIDENCE_CHANGED')
    mappings={r['sample_id']:r for r in json.loads(staticpath.read_text())['results']}
    for name,bound in design['fingerprint']['code'].items():
        if sha(ROOT/'g1_pipeline'/f'{name}.py')!=bound:raise ValueError('SOURCE_RUN_CODE_CHANGED:'+name)
    ids={r['sample_id'] for r in rows};measured={r['sample_id']:r for r in records}
    if len(rows)!=100 or len(ids)!=100 or len({r['source_family_id'] for r in rows})!=100:raise ValueError('INVALID_FIXED_SOURCE_PANEL')
    if len(records)!=100 or len(measured)!=100 or set(measured)!=ids:issues.append('FIXED100_INCOMPLETE')
    evidence={str(p.relative_to(ROOT)):sha(p) for p in [manifest,ledger,base/'run.json',PATH,staticpath]}
    if source=='hiphi':
        package=ROOT/'audit/hiphi_package_evidence_v1.json'
        if sha(package)!=design['fingerprint']['package_evidence']:raise ValueError('SOURCE_PACKAGE_EVIDENCE_CHANGED')
        evidence[str(package.relative_to(ROOT))]=sha(package)
    videopath=ROOT/'videos'/namespace/'manifest.json';video={}
    if videopath.exists():
        evidence[str(videopath.relative_to(ROOT))]=sha(videopath)
        for item in json.loads(videopath.read_text())['videos']:
            for key,bound in [('path','sha256'),('sheet_path','sheet_sha256')]:
                path=ROOT/item[key]
                if sha(path)!=item[bound]:raise ValueError('VIDEO_CHANGED')
                evidence[str(path.relative_to(ROOT))]=item[bound]
            for path,bound in item['input_hashes'].items():
                if sha(ROOT/path)!=bound:raise ValueError('VIDEO_TRAJECTORY_CHANGED')
            if item['full_clip'] and item['fps']==25:video[item['sample_id']]=item
    selection=select_videos(namespace)
    if len(video)<20 or not set(selection)<=set(video):issues.append('REQUIRED_STRATIFIED_AND_ANOMALY_VIDEOS_MISSING')
    cfg=yaml.safe_load((ROOT/policy['gate_config']).read_text());kin=cfg['kinematic'];reviewpath=ROOT/'audit'/f'{namespace}_review_decisions.json'
    decisions={};review={}
    if reviewpath.exists():
        review=json.loads(reviewpath.read_text())
        if review.get('policy_sha256')!=sha(PATH):raise ValueError('SOURCE_REVIEW_POLICY_CHANGED')
        for path,bound in review.get('bindings',{}).items():
            if sha(ROOT/path)!=bound:raise ValueError('SOURCE_REVIEW_EVIDENCE_CHANGED')
        if not evidence.items()<=review.get('bindings',{}).items():raise ValueError('UNBOUND_SOURCE_REVIEW_EVIDENCE')
        decisions={r['sample_id']:r for r in review['decisions']}
        if review.get('systematic_checks')!={key:'PASS' for key in ('left_right','scale','time','joint_order','native_geometry','support_scene')} or not review.get('systematic_check_rationale'):
            issues.append('SOURCE_SYSTEMATIC_REVIEW_INCOMPLETE')
    else:issues.append('SOURCE_SYSTEMATIC_REVIEW_MISSING')
    results=[];groups=defaultdict(lambda:Counter());numbers=defaultdict(list);pending=[];artifacts={}
    for meta in rows:
        sid=meta['sample_id'];r=measured.get(sid,{});reasons=list(r.get('rejections',[]));flags=list(r.get('review_flags',[]))
        if not r or r.get('status')=='ERROR':reasons.append('EXECUTION_ERROR')
        if r.get('output_paths'):
            if not r.get('output_hashes'):raise ValueError('MISSING_OUTPUT_HASHES')
            for path,bound in r['output_hashes'].items():
                if sha(ROOT/path)!=bound:raise ValueError('SOURCE_OUTPUT_CHANGED:'+path)
                artifacts[path]=bound
            mapping=ROOT/r['mapping_config']
            if sha(mapping)!=r['mapping_sha256']:raise ValueError('SOURCE_MAPPING_CHANGED')
            artifacts[str(mapping.relative_to(ROOT))]=r['mapping_sha256']
            mp=mappings[sid];mappingaudit=ROOT/mp['audit_path']
            if sha(mappingaudit)!=mp['audit_sha256']:raise ValueError('SOURCE_MAPPING_AUDIT_CHANGED')
            artifacts[str(mappingaudit.relative_to(ROOT))]=mp['audit_sha256']
            provenance=json.loads(mappingaudit.read_text())['source_provenance']
            native=HML/meta['source_paths']['native'] if source=='humanml3d' else ROOT/mp['native_path']
            if sha(native)!=r['source_sha256'] or sha(native)!=mp['native_sha256']:raise ValueError('NATIVE_SOURCE_CHANGED')
            artifacts[str(native)]=r['source_sha256']
            if source=='humanml3d':
                from pathlib import Path
                for path,bound in [(Path(provenance['native_model']),provenance['native_model_sha256']),
                                   (Path(provenance['native_model'].replace('/smplh/','/dmpls/')),provenance['dmpl_model_sha256'])]:
                    if sha(path)!=bound:raise ValueError('SOURCE_BODY_MODEL_CHANGED')
                    artifacts[str(path)]=bound
            c=json.loads((ROOT/r['output_paths']['contract_path']).read_text());m=json.loads((ROOT/r['metrics_path']).read_text())
            g=m['geometry'];cont=m['continuity']
            if c['status']!='PASS' or r.get('contract_status')!='PASS':reasons.append('NUMERICAL_CONTRACT')
            if g['ground_penetration_p95_m']>kin['ground_penetration_p95_max_m'] or g['ground_penetration_max_m']>kin['ground_penetration_max_m']:reasons.append('GROUND')
            if g['raw_self_collision_depth_over_1cm_frame_fraction']>kin['forbidden_self_collision_frame_fraction_max'] or g['raw_self_collision_max_depth_m']>kin['forbidden_self_collision_depth_max_m']:reasons.append('SELF_COLLISION')
            if cont['hinge_limit_excess_max_rad']>1e-4:reasons.append('HINGE_LIMIT')
            if any(v is not None and v>kin['contact_foot_skating_p95_max_m_s'] for v in c['skating_p95_m_s'].values()):reasons.append('SKATING')
            if not r.get('native_provenance_validated'):reasons.append('NATIVE_PROVENANCE')
            for key,value in g.items():
                if isinstance(value,(float,int)) and not isinstance(value,bool):numbers[key].append((value,r['frames']))
            if sid in video and video[sid]['frames']!=r['frames']:raise ValueError('VIDEO_DURATION_CHANGED')
        else:reasons.append('OUTPUT_MISSING')
        d=decisions.get(sid)
        if d and d.get('disposition')=='REJECT':reasons.append('REVIEW_REJECTED')
        if flags and not reasons:
            if not d:pending.append(sid)
            elif d.get('disposition')!='RETAIN_KINEMATIC_ONLY' or not d.get('rationale') or not d.get('visual_evidence'):raise ValueError('INVALID_SOURCE_REVIEW')
            elif any(path not in evidence for path in d['visual_evidence']):raise ValueError('UNBOUND_SOURCE_VISUAL_REVIEW')
        kept=not reasons and sid not in pending
        result={'sample_id':sid,'quality_pass':kept,'rejections':sorted(set(reasons)),'review_flags':flags,'review_pending':sid in pending}
        results.append(result)
        for kind,label in [('category',meta.get('category')),('fps',str(r.get('source_metadata',meta).get('original_fps'))),('scene',r.get('source_semantics',{}).get('scene_class'))]:
            group=groups[kind+':'+str(label)];group['input']+=1;group['retained']+=int(kept)
    accepted=sum(r['quality_pass'] for r in results)
    if accepted<95:issues.append('QUALITY_RETENTION_BELOW_95_OF_100')
    if pending:issues.append('ANOMALY_REVIEW_PENDING')
    aggregate={k:{'clip_macro_mean':float(np.mean([v for v,w in a])),'frame_weighted_mean_of_clip_metric':float(np.average([v for v,w in a],weights=[w for v,w in a])),
        'clip_median':float(np.median([v for v,w in a])),'clip_p95':float(np.percentile([v for v,w in a],95)),'clip_max':max(v for v,w in a)} for k,a in numbers.items()}
    report={'execution_scope':'isolated_pilot' if pilot else 'production','status':'PASS' if not issues else 'FAIL','source':source,'namespace':namespace,'expected':100,'attempted':len(records),
        'kinematic_retained':accepted,'rejected_or_pending':100-accepted,'failure_reasons':issues,'review_pending_ids':pending,
        'groups':dict(groups),'aggregate':aggregate,'results':results,'required_video_selection':selection,'review_required_bindings':evidence,
        'training_release':False,'dynamic_status':'NOT_IN_PILOT_SCOPE' if pilot else 'REQUIRED_AT_STAGE_4'}
    path=ROOT/'qa/aggregate'/f'{namespace}_acceptance.json';write_json(path,report)
    write_jsonl(base/'accepted.jsonl',[dict(measured[r['sample_id']],status='QA_PASS',training_release=False) for r in results if r['quality_pass']])
    write_jsonl(base/'rejected.jsonl',[r for r in results if not r['quality_pass']])
    # Bind just this stage and its actual shared execution dependencies. Later
    # independent source runners can be developed without invalidating it.
    code=set(k for k in policy['bindings'] if k.startswith('g1_pipeline/') and k.endswith('.py'))
    code.update('g1_pipeline/'+name+'.py' for name in design['fingerprint']['code'])
    code.update(['g1_pipeline/source_smoke_evaluation.py','g1_pipeline/render_frozen_review.py'])
    configs=[k for k in policy['bindings'] if k.startswith('configs/')]
    state=gate(stage,report['status'],';'.join(issues) or None,actual_quantity=len(records),metrics={'quality_pass':accepted,'denominator':100},
        input_paths=[manifest,ledger,base/'run.json',path,*([reviewpath] if reviewpath.exists() else []),*([videopath] if videopath.exists() else [])],
        fingerprint_scope={'code_paths':sorted(code),'config_paths':sorted(configs)},upstream_dependencies=[prior],
        supported_scope='source-evidenced flat ground and reviewed floor multisupport; no external furniture support')
    state['input_hashes'].update(artifacts)
    state['input_hashes'].update(evidence)
    state['input_manifest_hash']=digest(state['input_hashes']);write_json(ROOT/'gates'/('pilot' if pilot else '')/f'{stage}.json',state)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--namespace',required=True);p.add_argument('--source',required=True,choices=['humanml3d','hiphi']);p.add_argument('--render',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.render:
        from .render_frozen_review import run
        run(a.namespace,list(select_videos(a.namespace)),source=a.source)
    r=evaluate(a.namespace,a.source,pilot=a.pilot);print({k:r[k] for k in ('status','attempted','kinematic_retained','failure_reasons')})
