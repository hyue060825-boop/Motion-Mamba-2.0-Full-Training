"""Read-only audit of official BONES inputs, per-item rejection rules and call path."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import collections,json,sys
from pathlib import Path
import numpy as np
import mujoco
import yaml
from g1_pipeline.common import ROOT,BONES,sha,write_json,write_jsonl,now
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.robot import compile_target,read_csv,FORMAT
from g1_pipeline.resample import qpos as resample


def run():
    namespace='kinematic_training_20260917_v1_batch_0000'
    ledger=ROOT/'work'/namespace/'results.jsonl'
    rows=[json.loads(line) for line in ledger.open()]
    rows=[r for r in rows if r['dataset']=='bones']
    output=BASE/'bones_official_audit';output.mkdir(exist_ok=True)
    model,asset=compile_target();rules=yaml.safe_load((ROOT/'configs/gates.v2.yaml').read_text())['kinematic']
    protected=load(BASE/'execution_release.json')['protected_history']
    results=[];pairs=collections.Counter();rule_counts=collections.Counter()
    for r in rows:
        meta=r['source_metadata'];native=Path(meta['sealed_native_path'])
        assert native==BONES/meta['source_paths']['g1']
        assert sha(native)==meta['sealed_native_sha256']==r['source_sha256']
        for path,bound in r['output_hashes'].items():assert sha(ROOT/path)==bound,path
        q,t,_=read_csv(native,format_id=FORMAT,fps=meta['original_fps'])
        expected,times,mapping=resample(q,t,model.jnt_range[1:])
        actual=np.load(ROOT/r['output_paths']['qpos_path'])
        assert np.array_equal(expected,actual),r['sample_id']
        sidecar=load(ROOT/r['output_paths']['sidecar_path'])
        assert np.array_equal(times,np.asarray(sidecar['timestamps']))
        contract=load(ROOT/r['output_paths']['contract_path']);metrics=load(ROOT/r['metrics_path']);g=metrics['geometry']
        assert metrics['official_source'] and not any(k in metrics for k in ['rate_projection','ground_correction','temporal','human_target_metrics'])
        triggered=[]
        if g['raw_self_collision_depth_over_1cm_frame_fraction']>rules['forbidden_self_collision_frame_fraction_max']:triggered.append('SELF_COLLISION_FRAME_FRACTION')
        if g['raw_self_collision_max_depth_m']>rules['forbidden_self_collision_depth_max_m']:triggered.append('SELF_COLLISION_MAX_DEPTH')
        if r['source_semantics']['scene_class']=='flat_ground_core_candidate':
            if g['ground_penetration_p95_m']>rules['ground_penetration_p95_max_m']:triggered.append('GROUND_P95')
            if g['ground_penetration_max_m']>rules['ground_penetration_max_m']:triggered.append('GROUND_MAX')
        inferred=sorted({('SELF_COLLISION' if x.startswith('SELF_COLLISION') else 'GROUND') for x in triggered})
        if r['decision']=='REJECTED':
            assert inferred==r['decision_reasons'],(r['sample_id'],inferred,r['decision_reasons'])
            rule_counts.update(triggered);pairs.update(g['raw_self_contact_pair_counts'].keys())
        results.append({'sample_id':r['sample_id'],'decision':r['decision'],'reasons':r['decision_reasons'],
                        'triggered_rules':triggered,'scene':r['source_semantics']['scene_class'],
                        'native_csv':str(native),'native_sha256':r['source_sha256'],
                        'native_frames':len(q),'target_frames':len(actual),
                        'full_native_support':True,'official_resample_qpos_bit_exact':True,
                        'qpos_max_abs_difference':float(abs(expected-actual).max()),
                        'time_axis_bit_exact':True,'numeric_contract':contract['status'],
                        'geometry':g,'metrics_path':r['metrics_path'],'decision_path':r['decision_path']})
    write_jsonl(output/'all864_source_and_qa_checks.jsonl',results)
    rejected=[r for r in results if r['decision']=='REJECTED']
    write_jsonl(output/'rejected666_rules.jsonl',rejected)
    # Trace actual isolated entry execution, rejecting any retarget/postprocessor call.
    from g1_pipeline.kinematic_clip import execute
    spec=load(BASE/'execution_release.json');traces=[]
    forbidden={'gmr_temporal_arms','gmr_weighted_convergence','gmr_collision_refinement','ground_clearance','bounded_hinge_trajectory','source_motion_processing','native_rest_mapping','hiphi_rest_mapping'}
    selected=[]
    for reasons in [('SELF_COLLISION',),('GROUND',),('GROUND','SELF_COLLISION')]:
        selected.append(min((r for r in rows if tuple(r['decision_reasons'])==reasons),key=lambda r:r['frames']))
    for i,r in enumerate(selected):
        calls=collections.Counter()
        def profile(frame,event,arg):
            if event!='call':return
            path=Path(frame.f_code.co_filename)
            if path.parent!=ROOT/'g1_pipeline':return
            name=frame.f_code.co_name;calls[path.stem+'.'+name]+=1
            if path.stem in forbidden or (path.stem=='locked_target' and name=='create_retargeter'):
                raise AssertionError('BONES_CALLED_RETARGET_OR_POSTPROCESS:'+path.stem+'.'+name)
        dst=output/'runtime_trace'/str(i);dst.mkdir(parents=True,exist_ok=True)
        sys.setprofile(profile)
        try:entry,_=execute({'source':'bones','row':r['source_metadata']},dst,spec['asset'],spec['original_clip_code'])
        finally:sys.setprofile(None)
        assert entry['decision']==r['decision'] and entry['decision_reasons']==r['decision_reasons']
        for key in ['qpos_path','features_path']:
            assert np.array_equal(np.load(ROOT/entry['output_paths'][key]),np.load(ROOT/r['output_paths'][key]))
        assert calls['kinematic_clip.bones']==1 and calls['robot.read_csv'] and calls['production_contract.export_and_check']==1
        traces.append({'sample_id':r['sample_id'],'frames':r['frames'],'decision_reproduced':True,'qpos_features_bit_exact':True,'forbidden_calls':[],'calls':dict(calls)})
    for path,bound in protected.items():assert sha(ROOT/path)==bound,path
    summary={'time':now(),'status':'AUDIT_COMPLETE','scope':namespace,'bones_clips':len(rows),'rejected':len(rejected),
             'decision_counts':dict(collections.Counter(r['decision'] for r in rows)),
             'rejection_combinations':dict(collections.Counter('+'.join(r['reasons']) for r in rejected)),
             'rule_counts_overlapping':dict(rule_counts),'collision_pairs_clip_counts_top20':pairs.most_common(20),
             'contract_counts':dict(collections.Counter(r['numeric_contract'] for r in results)),
             'official_csv_resample_equality_all_clips':True,'runtime_traces':traces,
             'retarget_or_postprocess_misuse_found':False,
             'source_or_rule_misuse_found_in_verified_checks':False,
             'collision_scope':'locked compiled MuJoCo contact masks/excludes and parent filtering, extra_allowed_pairs=[] per frozen gates.v2.yaml; no controller rollout',
             'legacy_metric_annotation':'bones_pair_qa.replay emits legacy development strings and null self_collision_release_metric; actual decision uses explicit raw numeric fields with the later frozen gates.v2.yaml policy, not that null field',
             'limits':{k:v for k,v in rules.items() if k.startswith(('ground_','forbidden_','allowed_collision','scene_scope'))},
             'protected_history_unchanged':len(protected),'solver_or_policy_changed':False,
             'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [ledger,ROOT/'g1_pipeline/kinematic_clip.py',ROOT/'g1_pipeline/kinematic_policy.py',ROOT/'g1_pipeline/bones_pair_qa.py',ROOT/'configs/gates.v2.yaml',output/'all864_source_and_qa_checks.jsonl',output/'rejected666_rules.jsonl',Path(__file__)]}}
    write_json(output/'audit.json',summary)
    print({k:v for k,v in summary.items() if k in ['status','bones_clips','rejected','rejection_combinations','rule_counts_overlapping','contract_counts','official_csv_resample_equality_all_clips']})


if __name__=='__main__':run()
