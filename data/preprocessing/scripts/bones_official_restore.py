"""Publish official BONES qualifications from existing artifacts; no conversion calls."""
import collections,json
from g1_pipeline.common import ROOT,sha,write_json,write_jsonl,read_jsonl,now
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.bones_official_manifest_admission import apply,POLICY,ADDENDUM
from kinematic_finalize import record_function

def run():
    inputs=load(BASE/'input_release.json');family_splits=collections.defaultdict(set)
    metadata={}
    for r in read_jsonl(BASE/'candidates.jsonl'):
        family_splits[r['source_family_id']].add(r['project_split'])
        if r['dataset']=='bones':metadata[r['sample_id']]=r
    rows=[];bindings={}
    for i in range(len(inputs['panels'])):
        p=ROOT/'work'/f'kinematic_training_20260917_v1_batch_{i:04d}'/'results.jsonl'
        if not p.exists():continue
        # Atomic ledgers provide a stable snapshot; an active batch may advance later.
        data=p.read_bytes();bindings[str(p.relative_to(ROOT))]=__import__('hashlib').sha256(data).hexdigest()
        rows.extend(r for r in map(json.loads,data.splitlines()) if r['dataset']=='bones')
    assert len(rows)==len({r['sample_id'] for r in rows})
    changed=[];qualified=[];decisions=[];features_splits=collections.defaultdict(set)
    for r in rows:
        if r.get('output_paths'):
            h=r['output_hashes'][r['output_paths']['features_path']]
            features_splits[h].add(metadata[r['sample_id']]['project_split'])
    seen=set();record=record_function()
    for original in sorted(rows,key=lambda r:r['sample_id']):
        r=apply(original);meta=metadata[r['sample_id']]
        restored=original.get('decision') in ('REJECTED','REVIEW_PENDING') and r.get('decision')=='ACCEPTED_KINEMATIC'
        decision={'sample_id':r['sample_id'],'old_decision':original.get('decision',original['status']),
                  'old_reasons':original.get('decision_reasons',original.get('rejections',[])),
                  'decision':r.get('decision',r['status']),'project_split':meta['project_split'],
                  'training_role':meta['training_role'],'restored':restored,'frames':r.get('frames',0),
                  'decision_path':r.get('decision_path'),'training_included':False}
        if restored:changed.append(r['sample_id'])
        decisions.append(decision)
        if r.get('decision')!='ACCEPTED_KINEMATIC':continue
        feature_hash=r['output_hashes'][r['output_paths']['features_path']]
        reasons=[]
        if meta['training_role']=='held_out_calibration':reasons.append('FROZEN_CALIBRATION_HELDOUT')
        if len(family_splits[meta['source_family_id']])>1:reasons.append('CROSS_SPLIT_FAMILY')
        if len(features_splits[feature_hash])>1:reasons.append('CROSS_SPLIT_EXACT_FEATURE')
        if feature_hash in seen:reasons.append('EXACT_FEATURE_DUPLICATE')
        if reasons:decision['manifest_exclusions']=reasons;continue
        value=record(meta,dict(r,status='KINEMATIC_CANDIDATE'),pipeline_version='v25_official_bones_admission_20260918',
                     sim_coverage={'status':'NOT_RUN','controller_hash':None},
                     leakage={'status':'PASS','scope':'existing family splits and observed exact duplicates; final whole-release audit still required'})
        seen.add(feature_hash)
        value.update(kinematic_status='ACCEPTED_KINEMATIC',bones_official_policy_sha256=sha(POLICY),
                     geometry_validation='DIAGNOSTIC_ONLY',decision_path=r['decision_path'],
                     historical_policy_result=r.get('historical_policy_result'),training_role=meta['training_role'],
                     output_hashes=r['output_hashes'],diagnostic_tags=r.get('diagnostic_tags',[]))
        qualified.append(value);decision['training_included']=meta['project_split']=='train'
    out=POLICY.parent/'current';out.mkdir(exist_ok=True)
    for split in ['train','val','test']:write_jsonl(out/f'accepted_{split}.jsonl',[r for r in qualified if r['project_split']==split])
    write_jsonl(out/'all_decisions.jsonl',decisions)
    report={'time':now(),'status':'BONES_QUALIFICATIONS_RESTORED_PARTIAL_RELEASE','policy_sha256':sha(POLICY),
            'converted_bones_examined':len(rows),'restored_geometry_only_exclusions':sum(r['restored'] and r['old_decision']=='REJECTED' for r in decisions),
            'restored_geometry_scope_pending':sum(r['restored'] and r['old_decision']=='REVIEW_PENDING' for r in decisions),
            'scene_scope_policy_sha256':sha(ADDENDUM),
            'first_batch_666_restored':sum(s in set(changed) for s in [r['sample_id'] for r in read_jsonl(BASE/'bones_official_audit/rejected666_rules.jsonl')]),
            'manifest_counts':dict(collections.Counter(r['project_split'] for r in qualified)),
            'remaining_states':dict(collections.Counter(r['decision'] for r in decisions)),
            'manifest_exclusion_reasons':dict(collections.Counter(x for r in decisions for x in r.get('manifest_exclusions',[]))),
            'restored_train_included':sum(r['restored'] and r['training_included'] for r in decisions),
            'solver_reruns':0,'official_trajectories_modified':False,'original_decisions_rewritten':False,
            'final_training_ready':False,'scope':'converted official BONES only; val/test stay separate; heldout/family/exact duplicates excluded; final full-inventory manifests still required',
            'source_ledger_snapshot_hashes':bindings,'manifest_hashes':{s:sha(out/f'accepted_{s}.jsonl') for s in ['train','val','test']}}
    write_json(out/'report.json',report);print({k:v for k,v in report.items() if not k.endswith('hashes')})
    return report

if __name__=='__main__':run()
