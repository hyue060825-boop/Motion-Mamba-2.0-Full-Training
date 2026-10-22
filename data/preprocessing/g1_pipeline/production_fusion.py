"""Gated three-source comparison and frozen full-duration controller panel."""
import argparse
import json
import fcntl
from collections import defaultdict,Counter
import numpy as np
from .common import ROOT,BONES,read_jsonl,write_json,write_jsonl,sha,digest
from .gates import require_pass,gate
from .bones_scene_evidence import load as scenes
from .bones_relative_motion_v2 import labels
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample
from .production_contract import export_and_check
from .dynamic_tracking import Tracker

CLASSES=('locomotion','turning','crouching','upper_body','jumping','dancing_exercising','standing_idle','ground_body_support','balance_evasion')


def select_panel(records,per_source=30):
    chosen=[];source_counts={}
    for source in ('bones','humanml3d','hiphi'):
        source_rows=[r for r in records if r['dataset']==source];seen=set();selected={};coverage={}
        for label in CLASSES:
            candidates=sorted([r for r in source_rows if label in r['action_labels']],key=lambda r:digest([20260915,'dynamic',r['sample_id']]))
            coverage[label]=len({r['source_family_id'] for r in candidates})
            target=4 if label in CLASSES[:5] else 2
            for row in candidates:
                if target<=0:break
                if row['source_family_id'] in seen:continue
                seen.add(row['source_family_id']);selected[row['sample_id']]={**row,'matching_class':label};target-=1
        extras=sorted(source_rows,key=lambda r:digest([20260915,'dynamic_extra',r['sample_id']]))
        for row in extras:
            if len(selected)>=per_source:break
            if row['source_family_id'] in seen:continue
            label=next((k for k in CLASSES if k in row['action_labels']),None)
            if label is None:continue
            selected[row['sample_id']]={**row,'matching_class':label};seen.add(row['source_family_id'])
        if len(selected)<per_source:raise ValueError('INSUFFICIENT_INDEPENDENT_DYNAMIC_CLIPS:'+source)
        # Round-robin allocation has at most 28 before extras, so no selection
        # truncation can erase a requested core class.
        chosen.extend(list(selected.values())[:per_source]);source_counts[source]={'available':len(source_rows),'class_coverage':coverage,'selected':per_source}
    return chosen,source_counts


def baseline_candidates(panel='dev',namespace='bones_core_dev_v24'):
    manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl';meta={r['sample_id']:r for r in read_jsonl(manifest)}
    scene={r['sample_id']:r for r in scenes(manifest)}
    geometry=json.loads((ROOT/'work'/namespace/'pair_qa.json').read_text());candidates=[];rejected=[]
    for record in geometry['results']:
        sid=record['sample_id'];row=meta[sid];g=record.get('reference_geometry',{});sc=scene[sid]['scene']['scene_class'];reasons=[]
        if record.get('status')!='MEASURED' or record.get('reference_contract',{}).get('status')!='PASS':reasons.append('REFERENCE_NUMERICAL')
        if sc!='flat_ground_core_candidate':reasons.append('REFERENCE_SCENE_NOT_CORE')
        if not g or g['ground_penetration_p95_m']>.02 or g['ground_penetration_max_m']>.05:reasons.append('REFERENCE_GROUND')
        if not g or g['raw_self_collision_depth_over_1cm_frame_fraction']>.001 or g['raw_self_collision_max_depth_m']>.03:reasons.append('REFERENCE_SELF_COLLISION')
        if any(v is not None and v>.2 for v in record.get('reference_skating_p95_m_s',{}).values()):reasons.append('REFERENCE_SKATING')
        action=labels({**row,'source_description':scene[sid]['technical_description']})
        if not action:reasons.append('REFERENCE_ACTION_UNSUPPORTED')
        if reasons:rejected.append({'sample_id':sid,'reasons':reasons});continue
        source=BONES/row['source_paths']['g1']
        candidates.append({'sample_id':sid,'dataset':'bones','source_family_id':row['source_family_id'],'source_metadata':row,
            'action_labels':action,'scene_class':sc,'reference_path':str(source),'reference_sha256':sha(source),
            'original_fps':row['original_fps'],'gmr_role':'official_g1_main_pool','frames':record['frames']})
    return candidates,rejected


def source_candidates(source):
    current=ROOT/'audit'/f'current_{"hml" if source=="humanml3d" else "hiphi"}100_run.json';namespace=json.loads(current.read_text())['namespace']
    path=ROOT/'work'/namespace/'accepted.jsonl';records=read_jsonl(path)
    return [{**r,'dataset':source,'action_labels':r['source_semantics']['action_labels'],'scene_class':r['source_semantics']['scene_class'],
             'origin_namespace':namespace} for r in records]


def comparison(rows):
    groups=defaultdict(list)
    for row in rows:
        path=ROOT/row['output_paths']['qpos_path'];q=np.load(path);duration=(len(q)-1)/25
        speed=np.linalg.norm(np.diff(q[:,:3],axis=0)*25,axis=1)
        item={'sample_id':row['sample_id'],'duration_s':duration,'root_height_mean_m':float(q[:,2].mean()),
            'root_speed_mean_m_s':float(speed.mean()),'root_speed_p95_m_s':float(np.percentile(speed,95)),
            'joint_angle_mean_rad':q[:,7:].mean(axis=0).tolist(),'joint_angle_std_rad':q[:,7:].std(axis=0).tolist(),
            'joint_speed_p95_rad_s':np.percentile(abs(np.diff(q[:,7:],axis=0)*25),95,axis=0).tolist(),
            'matching_class':row['matching_class'],'qpos_sha256':sha(path)}
        groups[row['dataset']].append(item)
    return dict(groups)


def aggregate_dynamic(panel,tracks):
    byid=defaultdict(list)
    for r in tracks:byid[r['sample_id']].append(r)
    bysource=defaultdict(list);classes=defaultdict(lambda:defaultdict(list));incomplete=[]
    for row in panel:
        runs=byid[row['sample_id']]
        if len(runs)!=3 or {r['seed'] for r in runs}!={0,1,2}:incomplete.append(row['sample_id']);continue
        clip={'sample_id':row['sample_id'],'success':all(r['status']=='PASS' for r in runs),'seed_success_fraction':np.mean([r['status']=='PASS' for r in runs]),
              'joint_rmse_rad':float(np.mean([r['joint_rmse_rad'] for r in runs])),'matching_class':row['matching_class']}
        bysource[row['dataset']].append(clip);classes[row['matching_class']][row['dataset']].append(clip)
    summary={};problems=[]
    if incomplete:problems.append('INCOMPLETE_THREE_SEED_TRACKING')
    rng=np.random.default_rng(20260915)
    for source in ('bones','humanml3d','hiphi'):
        rows=bysource[source]
        if len(rows)<30:problems.append('SOURCE_PANEL_BELOW_30:'+source)
        if not rows:continue
        success=np.array([r['success'] for r in rows],float);rmse=np.array([r['joint_rmse_rad'] for r in rows]);indices=rng.integers(len(rows),size=(5000,len(rows)))
        summary[source]={'independent_clips':len(rows),'runs':len(rows)*3,'success_fraction_all3seeds':float(success.mean()),
            'mean_seed_success_fraction':float(np.mean([r['seed_success_fraction'] for r in rows])),
            'success_ci95_cluster_bootstrap':np.quantile(success[indices].mean(axis=1),[.025,.975]).tolist(),
            'joint_rmse_clip_macro_rad':float(rmse.mean()),'joint_rmse_ci95_cluster_bootstrap':np.quantile(rmse[indices].mean(axis=1),[.025,.975]).tolist()}
        if success.mean()<.9:problems.append('SOURCE_DYNAMIC_SUCCESS_BELOW_90_PERCENT:'+source)
    matched={}
    for source in ('humanml3d','hiphi'):
        keys=[k for k,g in classes.items() if g.get('bones') and g.get(source)];missing=[k for k,g in classes.items() if g.get(source) and not g.get('bones')]
        if missing:problems.append('MISSING_MATCHED_BONES_CLASSES:'+source+':'+','.join(missing))
        if not keys:problems.append('NO_MATCHED_DYNAMIC_ACTIONS:'+source);continue
        values=[]
        for k in keys:
            a=classes[k][source];b=classes[k]['bones'];v={'class':k,'source_n':len(a),'bones_n':len(b),
                'source_success':float(np.mean([r['success'] for r in a])),'bones_success':float(np.mean([r['success'] for r in b])),
                'source_joint_rmse':float(np.mean([r['joint_rmse_rad'] for r in a])),'bones_joint_rmse':float(np.mean([r['joint_rmse_rad'] for r in b]))};values.append(v)
        decline=float(np.mean([r['bones_success']-r['source_success'] for r in values]));ratio_excess=float(np.mean([r['source_joint_rmse']-(1.25*r['bones_joint_rmse']+.02) for r in values]))
        matched[source]={'action_equal_weighted_success_decline':decline,'action_equal_weighted_joint_rmse_bound_excess_rad':ratio_excess,'classes':values}
        if decline>.05+1e-12:problems.append('MATCHED_SUCCESS_DECLINE:'+source)
        if ratio_excess>1e-12:problems.append('MATCHED_JOINT_TRACKING_BOUND:'+source)
    return {'status':'PASS' if not problems else 'FAIL','failure_reasons':problems,'source_summary':summary,'matched':matched,
        'incomplete_ids':incomplete,'uncertainty_method':'5000 seeded bootstrap replicates resampling whole clips; all3seeds remain in a cluster',
        'limitation':'30 independent clips per source gives limited class coverage; no hardware or real-robot claim'}


def run(*,resume=False,dry_run=False,workers=1,max_clips=None,seed=20260915):
    require_pass('03_hiphi100')
    if workers!=1 or seed!=20260915 or max_clips is not None:raise ValueError('FIXED_FUSION_RESOURCE_AND_PANEL')
    bones,rejected=baseline_candidates()
    independent,independent_rejected=baseline_candidates('accept','bones_frozen_accept_v25')
    bones+=independent;rejected+=independent_rejected
    hml=source_candidates('humanml3d');hiphi=source_candidates('hiphi');panel,coverage=select_panel(bones+hml+hiphi)
    protocol=ROOT/'configs/dynamic_protocol.v1.json';fingerprint={'protocol':sha(protocol),'upstream':sha(ROOT/'gates/03_hiphi100.json'),
        'panel':digest(panel),'module':sha(__file__)};namespace='fusion_v1_'+digest(fingerprint)[:12];out=ROOT/'work'/namespace
    if dry_run:return {'namespace':namespace,'selected':len(panel),'tracking_runs':len(panel)*3,'coverage':coverage,'resources':{'workers':1,'threads':1,'gpu':False}}
    out.mkdir(exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('FUSION_ALREADY_RUNNING')
        design={'namespace':namespace,'fingerprint':fingerprint,'coverage':coverage,'source_input_denominators':{'bones_dev':200,'bones_accept':300,'humanml3d':100,'hiphi':100}}
        if (out/'run.json').exists():
            if json.loads((out/'run.json').read_text())!=design or not resume:raise ValueError('FUSION_RESUME_DESIGN_MISMATCH')
        write_json(out/'run.json',design);write_jsonl(out/'selected_inputs.jsonl',panel);write_jsonl(out/'bones_reference_rejected.jsonl',rejected)
        model,_=compile_target();prepared=[]
        for row in panel:
            if row['dataset']=='bones':
                native,nt,_=read_csv(row['reference_path'],format_id=FORMAT,fps=row['original_fps']);q,t,_=resample(native,nt,model.jnt_range[1:])
                directory=out/'bones'/row['sample_id'].split(':',1)[1]
                c,paths,hashes=export_and_check(directory,row['sample_id'],q,t,{'source':'official BONES G1 main pool','source_sha256':row['reference_sha256']},row['source_metadata']['captions'])
                if c['status']!='PASS':raise ValueError('DYNAMIC_OFFICIAL_BONES_NUMERIC_FAILURE')
                row={**row,'output_paths':paths,'output_hashes':hashes,'frames':len(q)}
            prepared.append(row)
        write_jsonl(out/'sim_panel.jsonl',prepared)
        write_json(ROOT/'qa/aggregate'/f'{namespace}_kinematic_distributions.json',comparison(prepared))
        from .motion_duplicates import compare as duplicates
        duplicates([{**r,'qpos_path':r['output_paths']['qpos_path'],'project_split':r['source_metadata']['project_split'],
                     'source_sha256':r.get('source_sha256',r.get('reference_sha256'))} for r in prepared],
                   ROOT/'qa/aggregate'/f'{namespace}_duplicate_candidates.json')
        tracker=Tracker(protocol);ledger=out/'tracking_results.jsonl';tracks=read_jsonl(ledger) if ledger.exists() else [];done={(r['sample_id'],r['seed']):r for r in tracks}
        gate('04_fusion_qa','RUNNING',input_paths=[out/'run.json',out/'sim_panel.jsonl',protocol],upstream_dependencies=['03_hiphi100'])
        for row in prepared:
            q=np.load(ROOT/row['output_paths']['qpos_path']);t=np.arange(len(q))*.04
            for runseed in (0,1,2):
                old=done.get((row['sample_id'],runseed));qp=out/'tracks'/(row['sample_id'].replace(':','__')+f'.seed{runseed}.npz')
                if resume and old and qp.exists() and sha(qp)==old['trajectory_sha256']:continue
                result=tracker.track(q,t,seed=runseed,scene_class=row['scene_class'],output=qp)
                result.update(sample_id=row['sample_id'],dataset=row['dataset'],source_family_id=row['source_family_id'],matching_class=row['matching_class'])
                done[(row['sample_id'],runseed)]=result;tracks=list(done.values());write_jsonl(ledger,tracks);print(row['sample_id'],runseed,result['status'],flush=True)
        dynamic=aggregate_dynamic(prepared,tracks);path=ROOT/'qa/aggregate'/f'{namespace}_dynamic.json';write_json(path,dynamic)
        report={'status':'FUSION_MEASURED_PENDING_RELEASE_REVIEW','namespace':namespace,'dynamic':dynamic,
            'source_counts':{'bones':len(bones),'humanml3d':len(hml),'hiphi':len(hiphi)},
            'required_remaining':['bound common-action/duration/speed/source systematic review','near-duplicate motion review','stats/tokenizer/sampler and full scope freeze'],
            'full_batch_authorized':False}
        write_json(ROOT/'qa/aggregate'/f'{namespace}.json',report);write_json(ROOT/'audit/current_fusion_run.json',{'namespace':namespace})
        gate('04_fusion_qa','BLOCKED',';'.join(dynamic['failure_reasons']+report['required_remaining']),actual_quantity=len(tracks),metrics=dynamic['source_summary'],
            input_paths=[out/'run.json',out/'sim_panel.jsonl',ledger,path,protocol],upstream_dependencies=['03_hiphi100'])
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args();print(run(resume=a.resume,dry_run=a.dry_run))
