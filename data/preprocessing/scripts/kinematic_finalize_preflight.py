"""Exercise the offline-only manifest and train statistics branches on real clips."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import copy
from g1_pipeline.common import ROOT,read_jsonl,write_json,write_jsonl,sha
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline import production_manifest
import kinematic_finalize as final


def run():
    destination=BASE/'runtime_preflight/finalization'
    original=read_jsonl(BASE/'runtime_preflight/manifest.jsonl')
    raws=load(BASE/'runtime_preflight/source_results.json')['results']
    branch=final.record_function()
    rebuilt=[]
    for meta,result in zip(original,raws):
        result=dict(result,status='KINEMATIC_CANDIDATE')
        kwargs=dict(pipeline_version='v25_kinematic_finalization_preflight',
                    sim_coverage={'status':'NOT_RUN','controller_hash':None},
                    leakage={'status':'PASS','scope':'non-final preflight'})
        try:production_manifest.record(meta,result,**kwargs)
        except ValueError as exc:assert str(exc)=='MANIFEST_DYNAMIC_COVERAGE_REQUIRED'
        else:raise AssertionError('ORIGINAL_PRODUCTION_GUARD_WAS_DISABLED')
        rebuilt.append(branch(meta,result,**kwargs))
        try:branch(meta,dict(result,contract_status='FAIL'),**kwargs)
        except ValueError as exc:assert str(exc)=='MANIFEST_REQUIRES_VALIDATED_OUTPUT'
        else:raise AssertionError('HARD_CONTRACT_FAILURE_ACCEPTED')
    final.BASE=destination
    path=destination/'manifests/accepted_train.jsonl'
    write_jsonl(path,rebuilt);stats=final.stats()
    assert stats['merged_frame_weighted']['frames']==sum(r['target_frames'] for r in rebuilt)
    assert stats['train_manifest_sha256']==sha(path) and not stats['normalizer_replaced']
    duplicate=dict(rebuilt[0],sample_id=rebuilt[0]['sample_id']+':preflight_duplicate')
    write_jsonl(path,rebuilt+[duplicate]);dedup=final.stats()
    assert dedup['merged_frame_weighted']['frames']==stats['merged_frame_weighted']['frames']
    assert len(dedup['exact_duplicate_exclusions'])==1
    for key,value in [('project_split','val'),('status','REJECTED')]:
        invalid=copy.deepcopy(rebuilt);invalid[0][key]=value;write_jsonl(path,invalid)
        try:final.stats()
        except ValueError as exc:assert str(exc)=='STATS_TRAIN_QA_ONLY'
        else:raise AssertionError('INVALID_STATS_ROW_ACCEPTED')
    write_jsonl(path,rebuilt);stats=final.stats()
    report={'status':'PASS','scope':'non-final real-three-source branch preflight',
            'dynamic_not_run_allowed_only_in_new_branch':True,
            'original_production_dynamic_guard_preserved':True,
            'hard_contract_negative_rejected':True,'validation_and_rejected_rows_excluded':True,
            'exact_duplicate_frames_excluded':True,'valid_frames':stats['merged_frame_weighted']['frames'],
            'source_clips':stats['unique_motion_counts'],'normalizer_replaced':False,
            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [path,ROOT/'scripts/kinematic_finalize.py',ROOT/'g1_pipeline/production_manifest.py',ROOT/'g1_pipeline/fusion_statistics.py']}}
    write_json(destination/'acceptance.json',report);print(report)


if __name__=='__main__':run()
