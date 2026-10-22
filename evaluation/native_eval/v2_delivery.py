"""Truthful capability gates, immutable assets, and executable offline smoke."""
from motion_mamba2.paths import artifact_path
import shutil
import time
import numpy as np
from .common import *
from .v2_prepare import DOMAINS
from .v2_score import score,verify_release

TRACKS={'humanml3d':'HumanML3D-G1-native-v2','bones':'BONES-G1-native-v2'}

def capability(state,scope,reason,**evidence):return dict(status=state,scope=scope,reason=reason,evidence=evidence)

def capabilities(d,results):
    tests=read(ROOT/'IMPLEMENTATION_TESTS_V2.json');policy=read(ROOT/'VALIDATION_POLICY_V2.json');vals=list(results.values());audit=read(ROOT/'SPLIT_AUDIT_V2.json')[d]
    numeric=tests['passed'] and all(tests['code'].get('native_eval/'+name)==sha(CODE/'native_eval'/name) for name in ('model.py','schema.py','encoding.py','metrics.py','v2_score.py','v2_statistics.py'))
    r1=[r['retrieval']['4101']['R@1'] for r in vals];gap=abs(r1[0]-r1[1]);stable=gap<=.15
    retrieval_ok=stable and all(r['paired_minus_shuffled']['ci95'][0]>0 and r['paired_minus_shuffled']['mean']>=.05 and r['retrieval']['4101']['R@1']>=.1 and r['length_matched']['N']>=32 and r['length_matched']['R1']>.1 for r in vals)
    mirrors=[r['mirrors'] for r in vals];direction=all(m['N']>=32 and m['margin']['ci95'][0]>0 for m in mirrors)
    def robust(dist,k):
        x=dist['variants'][k]
        return x['basic_ci95'][0]>0 and x['percentile_ci95'][0]>0 and x['positive_splits']==5
    sufficient=all(r['distribution']['N']>=1024 and r['distribution']['side_health']['a']['samples']>=256 and r['distribution']['health']['effective_rank']>=8 and r['distribution']['health']['mean_variance']>.001 for r in vals)
    distributions=[r['distribution'] for r in vals]
    if d=='bones':distributions += [r['actor_cluster_distribution'] for r in vals]
    short=numeric and sufficient and all(robust(dist,k) for dist in distributions for k in ('static','repeat4'))
    local=numeric and sufficient and all(robust(dist,k) for dist in distributions for k in ('shuffle4','shuffle52'))
    # FAIL means a well-sampled median effect is nonpositive on most splits;
    # uncertain intervals/cluster evidence remain INSUFFICIENT_EVIDENCE.
    short_state='PASS' if short else 'FAIL' if sufficient and any(dist['variants'][k]['positive_splits']<=1 for dist in distributions for k in ('static','repeat4')) else 'INSUFFICIENT_EVIDENCE'
    local_state='PASS' if local else 'FAIL' if sufficient and any(dist['variants'][k]['positive_splits']<=1 for dist in distributions for k in ('shuffle4','shuffle52')) else 'INSUFFICIENT_EVIDENCE'
    return dict(numeric_correctness=capability('PASS' if numeric else 'FAIL','formula, padding, precision, roles and release implementation','Regression tests and per-release smoke required',tests_file='IMPLEMENTATION_TESTS_V2.json'),
        basic_text_motion_retrieval=capability('PASS' if retrieval_ok and numeric else 'FAIL','full source semantic segments; fixed text->motion 32-candidate panels','paired-vs-shuffled effect, length matching and two training seeds',R1=r1,seed_gap=gap),
        direction_discrimination=capability('PASS' if direction else 'INSUFFICIENT_EVIDENCE','source-backed mirrored left/right controls','96-pair cap; joint reflection and matched text; not arbitrary direction semantics',panels=[dict(N=m['N'],margin=m['margin']) for m in mirrors]),
        event_order_discrimination=capability('INSUFFICIENT_EVIDENCE','multi-event A->B versus B->A','all captions scanned; aggregate pose evidence insufficient to verify temporal counterfactual; no reversal-as-order claim',panel=read(ROOT/'panels/SEMANTIC_PANELS.json')[d+':holdout']),
        short_window_distribution=capability(short_state,'center52, 256D FP64 native FID, fixed finite-panel comparison','Requires static and repeat4 effect stable across 5 splits, two seeds and BONES actor-cluster sensitivity. Bootstrap intervals descriptive; no population nominal-coverage guarantee.',sample_counts=[v['distribution']['N'] for v in vals],sufficient_sample_rule=sufficient),
        local_temporal_sensitivity=capability(local_state,'velocity-refreshed within4/within52 edits','Both strengths reported; sensitivity is not semantic event-order understanding'),
        full_sequence_distribution=capability('NOT_TESTED','continuous 9/20/40 second distribution','No independent continuous long-sequence reference validation; no renaming window averages as full FID'),
        data_provenance=capability('PASS' if not audit['family_overlap'] and not audit['actual_feature_overlap'] and not audit['canonical_duplicate_overlap'] else 'FAIL','audited admitted motion identities and v2 train-only statistics','source/canonical exact duplicates checked; unknown semantic near-duplicates and HumanML3D actor correlation disclosed'),
        generator_test_exposure=capability('INSUFFICIENT_EVIDENCE','generator generalization','v2 encoder holdout drawn from original train; historical generator and frozen-asset exposure not absent'))

def freeze(d,caps,selection):
    dest=ROOT/'releases'/TRACKS[d];existing=dest/'RELEASE.json'
    if existing.exists():return verify_release(dest)
    dest.mkdir(parents=True,exist_ok=True)
    for metric,head in [('retrieval','retrieval'),('distribution','distribution')]:
        s=selection[metric];src=ROOT/'runs'/s['variant']/d/'seed1701'/(s['head']+'.pt');shutil.copy2(src,dest/(head+'.pt'))
    for name in ('SKELETON.json','VALIDATION_POLICY_V2.json','SPLIT_AUDIT_V2.json','TEXT_REUSE_AUDIT.json','TEXT_BACKBONE_ACTUAL_HASHES.json','EVAL_CONTRACT_V2.json','TRAIN_PLAN_V2.json','DATA_MANIFEST_V2.json'):
        shutil.copy2(ROOT/name,dest/name)
    shutil.copy2(ROOT/'data'/f'{d}_stats.npz',dest/'stats.npz')
    qp=read(ROOT/'panels'/f'{d}_holdout.json');dp=read(ROOT/'panels'/f'{d}_holdout_distribution.json');qby={r['sample_id']:r for r in qp['records']}
    combined=dict(version='native-v2',records=[dict(r,caption=qby.get(r['sample_id'],r)['caption']) for r in dp['records']],retrieval_subset_ids=list(qby),distribution_window='center52');combined['panel_sha256']=identity(combined)
    fp=dict(version='native-v2-FID52',records=[dict(r,frames=52,reference_source_interval=[(r['frames']-52)//2,(r['frames']-52)//2+52]) for r in dp['records']]);fp['panel_sha256']=identity(fp)
    write(dest/'retrieval_panel.json',qp);write(dest/'combined_panel.json',combined);write(dest/'fid52_panel.json',fp)
    a=np.load(ROOT/'validation'/d/'seed1701/holdout/embeddings.npz')
    np.savez(dest/'reference_embeddings.npz',h52=a['h52'],text=a['text'],motion=a['motion'],distribution_ids=a['ids'],retrieval_ids=a['retrieval_ids'])
    for p in (CODE/'native_eval').glob('*.py'):
        target=dest/'code/native_eval'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    ref=np.asarray(a['h52'],np.float64);np.savez(dest/'reference_statistics.npz',mean=ref.mean(0),cov=np.cov(ref,rowvar=False),N=len(ref))
    release=dict(version='native-v2',track=TRACKS[d],domain=d,capabilities=caps,primary_seed=1701,sensitivity_seed=2903,selection=selection,reference_scope='evaluator encoder holdout from original train; not generator unseen test',inference='FP32 length-sorted batch32 restored by sample ID; standard FP64 FID',text='frozen google-t5/t5-3b, revision in TEXT_REUSE_AUDIT; query projected features frozen with retrieval checkpoint',status='READY_FOR_GENERATOR_INTEGRATION' if any(caps[k]['status']=='PASS' for k in ('basic_text_motion_retrieval','short_window_distribution')) else 'DIAGNOSTIC_ONLY',generator_integration='PENDING_NEW_GENERATOR_OUTPUT',files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
    release['release_sha256']=identity(release);write(existing,release);return release

def smoke(d,release):
    dest=ROOT/'smoke'/d;dest.mkdir(parents=True,exist_ok=True);folder=ROOT/'releases'/TRACKS[d];p=read(folder/'combined_panel.json');sk=read(folder/'SKELETON.json');metadata=[]
    # Explicit real-reference replay. Identities bind actual source manifest
    # and evaluator code, not an invented generator checkpoint.
    for r in p['records']:
        n=r['frames'];metadata.append(dict(schema_version='motion_eval_sample_v1',sample_id=r['sample_id'],dataset=d,track=TRACKS[d],source_family_id=r['source_family_id'],panel_id=p['panel_sha256'],checkpoint_sha256=sha(ROOT/'data'/f'{d}_holdout.jsonl'),generator_code_snapshot=identity(release['files']),seed=0,representation='physical414',skeleton='G1Skeleton34',skeleton_sha256=sk['sha256'],fps=25,units='meters',array_path=r['feature_path'],array_sha256=r['actual_feature_sha256'],array_key=None,array_layout='generated_only',coordinate_frame='world',up_axis='y',normalized=False,normalizer=None,stored_frames=n,generated_frame_intervals=[[0,n]],history_context_frames=104,history_in_array=False,padding_frame_intervals=[],preroll_frame_intervals=[],frame_roles=[dict(interval=[0,n],role='generated')],caption_intervals=[dict(start=0,end=n,text=r['caption'],alignment='full_semantic_segment')],conditioning=dict(mode='real_reference_interface_replay_NOT_GENERATOR',future_constraints=[],priming_receipt='no model; schema compatibility replay'),postprocess='none',completion_status='complete',source_time_mapping=dict(source_start_seconds=r['native_start_sec']),generation_time_mapping=dict(stored_frame0_generated_frame=0,purpose='real reference replay'),generation_receipt=dict(H=104,F=0,sampling_steps=1,cfg=1.,history_type='not_in_array',continuous_generation=True,resets=[],origin='REAL_REFERENCE_REPLAY_NOT_GENERATOR')))
    path=dest/'real_reference_replay.jsonl';jsonl(path,metadata)
    result=score(folder,path,dest/'SCORE.json','both',True,'cpu')
    if result['FID_native52_center_v2']>1e-4:raise ValueError('Real replay FID mismatch')
    expected=read(ROOT/'validation'/d/'seed1701/holdout/RESULT.json')['retrieval']['4101']['R@1']
    if abs(result['R_T2M_pool32']['4101']['R@1']-expected)>1/len(read(folder/'retrieval_panel.json')['records']):raise ValueError('Real replay retrieval identity/order mismatch')
    return dict(passed=True,N=result['N'],self_fid=result['FID_native52_center_v2'],retrieval_R1=result['R_T2M_pool32']['4101']['R@1'],purpose='full offline real interface check, not a generator score')

def generator_import_check():
    records=[]
    for r in rows(artifact_path('imported_generator_smoke_jsonl_52833b3b01')):
        path=Path(r['array_path'])
        if sha(path)!=r['array_sha256']:raise ValueError('Completed generator output changed')
        with np.load(path) as pack:
            a=pack['physical'];records.append(dict(sample_id=r['sample_id'],shape=list(a.shape),finite=bool(np.isfinite(a).all()),frames=len(a),declared_generated_frames=r['generated_frames'],history_context_frames=r['history_context_frames'],recropped_history=False,status='IMPORT_ONLY_METADATA_INCOMPLETE',missing=r['missing']))
    write(ROOT/'EXISTING_GENERATOR_IMPORT_CHECK.json',records);return records

def deliver():
    results=read(ROOT/'VALIDATION_RESULTS_V2.json');sel=read(ROOT/'SELECTION_V2.json');caps={d:capabilities(d,results[d]) for d in DOMAINS};write(ROOT/'CAPABILITIES.json',caps)
    releases={d:freeze(d,caps[d],sel['selected'][d]) for d in DOMAINS};smokes={d:smoke(d,releases[d]) for d in DOMAINS};write(ROOT/'OFFLINE_SCORE_SMOKE.json',smokes);generator_import_check()
    baseline=read(ROOT/'V1_READONLY_BASELINE.json');changed=[p for p,h in baseline.items() if sha(p)!=h];write(ROOT/'V1_READONLY_VERIFICATION.json',dict(changed=changed,checked=len(baseline)))
    if changed:raise ValueError('v1 read-only verification changed')
    from .v2_report import report
    report(results,caps,sel,releases,smokes)
    status('COMPLETE_WITH_CAPABILITY_LIMITATIONS',releases={d:r['status'] for d,r in releases.items()},generator_integration='PENDING_NEW_GENERATOR_OUTPUT',v1_unchanged=True,gpu_budget=read(ROOT/'GPU_BUDGET.json'),capabilities=caps)
    return dict(releases={d:r['status'] for d,r in releases.items()},smokes=smokes)
