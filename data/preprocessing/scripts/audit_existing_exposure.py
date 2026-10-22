"""Metadata overlap against existing project datasets; never imports old features."""
from motion_mamba2.paths import artifact_path
import json
from pathlib import Path
from collections import Counter,defaultdict
from g1_pipeline.common import ROOT,BONES,read_jsonl,write_json,sha


def run():
    paths={s:ROOT/'manifests'/p for s,p in [('bones','bones_lineage.jsonl'),('humanml3d','humanml3d_lineage.v2.jsonl'),('hiphi','hiphi_lineage.jsonl')]}
    source={s:read_jsonl(p) for s,p in paths.items()};byid={s:{r['source_motion_id']:r for r in rows} for s,rows in source.items()}
    bonespaths={str(BONES/r['source_paths']['g1']):r for r in source['bones']}
    reports=[];bindings={str(p.relative_to(ROOT)):sha(p) for p in paths.values()};exposure=defaultdict(set)
    for name in ('bones_seed','humanml3d'):
        dataset='bones' if name=='bones_seed' else name
        for split in ('train','val','test'):
            path=Path(artifact_path('result_d22902d183'))/f'{name}_motion_mamba2_{split}'/'manifest.jsonl'
            if not path.exists():continue
            bindings[str(path)]=sha(path);rows=read_jsonl(path);count=Counter();mismatches=[];unmapped=[]
            for old in rows:
                current=bonespaths.get(old.get('source_path')) if dataset=='bones' else byid[dataset].get(old['id'])
                if current is None:unmapped.append(old['id']);continue
                count[current['project_split'] or 'unknown']+=1;exposure[current['source_family_id']].add(split)
                if current['project_split']!=split:mismatches.append({'old_id':old['id'],'current_id':current['sample_id'],'existing_split':split,'current_split':current['project_split']})
            reports.append({'manifest':str(path),'dataset':dataset,'existing_split':split,'records':len(rows),'mapped_current_split_counts':dict(count),
                'split_mismatches':mismatches,'unmapped_count':len(unmapped),'unmapped_ids':unmapped})
    train_eval=[]
    for dataset,rows in source.items():
        for r in rows:
            if r['project_split']=='train' and r['eligibility']=='candidate' and exposure[r['source_family_id']]&{'val','test'}:
                train_eval.append({'sample_id':r['sample_id'],'source_family_id':r['source_family_id'],'prior_evaluation_splits':sorted(exposure[r['source_family_id']]&{'val','test'})})
    report={'status':'METADATA_EXPOSURE_AUDIT_ONLY','existing_manifests':reports,'eligible_train_overlap_prior_eval_families':train_eval,
        'eligible_train_overlap_prior_eval_count':len(train_eval),'bindings':bindings,
        'old_feature_cache_used':False,'cross_source_actor_identity':'UNKNOWN; namespaced IDs do not establish real-person disjointness',
        'cross_source_near_duplicate_motion':'PENDING_PHYSICAL_FINGERPRINT_REVIEW',
        'exposure_roles':{
            'gmr_adapter':'BONES fixed smoke16/dev200; accept300 frozen and actor-disjoint; known source IDs bound in calibration protocol',
            'native_human_adapter':'fixed source smoke input geometry; no official val/test fitting',
            'project_g1_generation_checkpoint':'known existing BONES train manifest and project-normalizer exposure; same-domain train reuse is disclosed',
            'released_g1_tokenizer':'UNKNOWN full pretraining list; published model cache has no complete per-take exposure manifest',
            'sonic_low_level_controller':'UNKNOWN full pretraining list; source repository/model hashes known',
            'text_encoder':'pretrained general language model; exact training text exposure unknown',
            'human_benchmark':'existing HumanML Core27 pipeline remains independent; feature caches are not imported'},
        'unqualified_no_leakage_claim':False,'training_release':False,'module_sha256':sha(__file__)}
    write_json(ROOT/'audit/existing_manifest_exposure_v1.json',report)
    print('existing manifest sets',len(reports),'eligible train / prior eval conflicts',len(train_eval),flush=True)
    return report


if __name__=='__main__':run()
