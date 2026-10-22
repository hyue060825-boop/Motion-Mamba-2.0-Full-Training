"""Resumable frozen SOMA/GMR calibration. Acceptance never fits parameters."""
import argparse
import io
import json
import shutil
import tarfile
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,write_jsonl,atomic_bytes,sha,now
from .calibration_protocol import require_frozen,PATH
from .gates import gate
from .locked_target import prepare,create_retargeter
from .gmr_soma import parse
from .gmr_temporal_arms import solve
from .gmr_collision_refinement import refine
from .ground_clearance import correct
from .bounded_hinge_trajectory import project,sonic_training_reference_limits
from .bones_scene_evidence import load as load_scenes
from .bones_continuity_qa import metrics as continuity
from .initial_planar_alignment import initial_alignment
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample,time_grid
from .bones_pair_qa import replay


def save_array(path,value):
    buf=io.BytesIO();np.save(buf,value);atomic_bytes(path,buf.getvalue())


def materialize(rows,panel):
    output=ROOT/'work'/('gmr_soma_smoke' if panel=='smoke' else 'bones_pairs_'+panel)
    output.mkdir(exist_ok=True);wanted={r['source_paths']['soma_uniform']:r for r in rows};results=[]
    for member,row in list(wanted.items()):
        path=output/(row['source_motion_id']+'.bvh');side=path.with_suffix('.source.json')
        if path.exists() and side.exists():
            bound=json.loads(side.read_text())
            if bound.get('member')==member and bound.get('sha256')==sha(path):results.append(bound);wanted.pop(member)
    if wanted:
        with tarfile.open(BONES/'soma_uniform.tar.gz','r|gz') as archive:
            for member in archive:
                row=wanted.get(member.name)
                if row is None:continue
                if not member.isfile():raise ValueError('SOURCE_MEMBER_NOT_REGULAR')
                content=archive.extractfile(member).read();path=output/(row['source_motion_id']+'.bvh');atomic_bytes(path,content)
                bound={'sample_id':row['sample_id'],'member':member.name,'archive':str(BONES/'soma_uniform.tar.gz'),
                       'sha256':sha(path),'bytes':len(content),'path':str(path),'time':now()}
                write_json(path.with_suffix('.source.json'),bound);results.append(bound);wanted.pop(member.name)
                if len(results)%10==0:print('materialized',len(results),'/',len(rows),flush=True)
                if not wanted:break
    if wanted:raise ValueError('MISSING_SOURCE_MEMBERS:'+str(list(wanted)))
    write_json(ROOT/f'audit/bones_frozen_{panel}_materialization.json',{'expected':len(rows),'results':results})
    return output


def target_errors(solver,frames,poses,cfg):
    names=list(cfg['ik_match_table2']);ids=[solver.model.body(b).id for b in names];human=[cfg['ik_match_table2'][b][0] for b in names]
    pe=[];oe=[]
    for frame,pose in zip(frames,poses):
        solver.update_targets(frame,offset_to_ground=False);solver.configuration.update(pose);d=solver.configuration.data
        target=solver.scaled_human_data
        pe.append(np.linalg.norm(d.xpos[ids]-np.array([target[n][0] for n in human]),axis=1))
        oe.append(np.rad2deg((R.from_matrix(d.xmat[ids].reshape(-1,3,3)).inv()*R.from_quat(np.array([target[n][1] for n in human]),scalar_first=True)).magnitude()))
    return np.array(pe),np.array(oe)


def run(panel='accept',resume=False,dry_run=False,verification=False):
    policy=require_frozen();manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl';rows=read_jsonl(manifest)
    if panel not in ('smoke','dev','accept'):raise ValueError('INVALID_PANEL')
    expected={'smoke':16,'dev':200,'accept':300}[panel]
    if len(rows)!=expected:raise ValueError('FIXED_PANEL_COUNT')
    if verification:
        if panel!='dev':raise ValueError('VERIFICATION_ONLY_FIXED_DEV')
        ids={'walk_forward_normal_002__A005','pels_air_punch_001__A493','dance_hiphop_kid_n_play_R_fast_002__A316'}
        rows=[r for r in rows if r['source_motion_id'] in ids]
        if len(rows)!=3:raise ValueError('VERIFICATION_SELECTION_MISSING')
    elif panel=='accept':
        p=ROOT/'audit/frozen_v25_reproduction.json'
        if not p.exists() or json.loads(p.read_text()).get('status')!='PASS' or json.loads(p.read_text()).get('policy_sha256')!=sha(PATH):
            raise RuntimeError('FROZEN_COMPOSITION_REPRODUCTION_REQUIRED')
    namespace=f'bones_frozen_{panel}_v25'+('_verification' if verification else '')
    out=ROOT/'work'/namespace
    if dry_run:return {'namespace':namespace,'selected':len(rows),'fixed_panel':expected,'resources':policy['resources'],'manifest_sha256':sha(manifest)}
    if shutil.disk_usage(ROOT).free<policy['resources']['stop_free_disk_below_gb']*1e9:raise RuntimeError('DISK_BUDGET_GUARD')
    design={'namespace':namespace,'panel':panel,'selected_sample_ids':[r['sample_id'] for r in rows],
            'policy_sha256':sha(PATH),'manifest_sha256':sha(manifest),'resources':policy['resources'],
            'full_clip':True,'reference_input_to_solver':False,'verification_only':verification}
    if (out/'run.json').exists():
        if json.loads((out/'run.json').read_text())!=design:raise ValueError('RUN_FINGERPRINT_CHANGED')
        if not resume:raise ValueError('EXISTING_NAMESPACE_REQUIRES_RESUME')
    out.mkdir(exist_ok=True);write_json(out/'run.json',design)
    if panel=='accept':gate('01_bones_calibration','RUNNING',actual_quantity=0,input_paths=[PATH,manifest],upstream_dependencies=['00_preflight'])
    source_root=materialize(rows,panel);scenes={r['sample_id']:r for r in load_scenes(manifest)}
    cfgpath=ROOT/policy['solver_config'];cfg=json.loads(cfgpath.read_text());post=json.loads((ROOT/policy['post_config']).read_text())
    xml,asset=prepare();model,_=compile_target();limits=sonic_training_reference_limits(model)
    cached={r['sample_id']:r for r in read_jsonl(out/'results.jsonl')} if (out/'results.jsonl').exists() else {};results=[]
    for row in rows:
        require_frozen();mid=row['source_motion_id'];sid=row['sample_id'];started=time.monotonic()
        qp=out/(mid+'.npy');tp=qp.with_suffix('.time.json');sp=qp.with_suffix('.solver.json');bp=out/(mid+'.unprocessed.npy')
        bvh=source_root/(mid+'.bvh');rp=BONES/row['source_paths']['g1'];old=cached.get(sid,{})
        bindings=[(qp,'output_sha256'),(tp,'time_sha256'),(sp,'solver_sha256'),(bp,'unprocessed_sha256'),(bvh,'source_sha256'),(rp,'reference_sha256')]
        if resume and old.get('status')=='DIAGNOSTIC_COMPLETE' and all(p.exists() and sha(p)==old.get(k) for p,k in bindings):
            saved=np.load(qp)
            if saved.shape==(old['frames'],36) and np.isfinite(saved).all():results.append(old);print(sid,'verified resume',flush=True);continue
        entry={'sample_id':sid,'status':'ERROR','retargeted':False,'reference_loaded':False}
        write_json(out/'heartbeat.json',{'sample_id':sid,'time':now(),'phase':'solving','attempt':old.get('attempt',0)+1})
        try:
            lines=bvh.read_text().splitlines();ix=lines.index('MOTION');n=int(lines[ix+1].split(':')[1]);dt=float(lines[ix+2].split(':')[1])
            reference,rt,_=read_csv(rp,format_id=FORMAT,fps=row['original_fps'])
            if n!=len(reference):raise ValueError('UNVERIFIED_PAIR_FRAME_COUNTS')
            end=np.nextafter(min((n-1)*dt,rt[-1]),np.inf);times,mapping=time_grid(np.arange(n)*dt,end=end)
            reference,tt,refmap=resample(reference,rt,model.jnt_range[1:],end=end)
            if not np.allclose(times,tt,atol=1e-10,rtol=0):raise ValueError('PAIRED_TIME_MISMATCH')
            frames,_=parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=times,rotation_basis='world_only')
            solver=create_retargeter(cfgpath,xml);q,temporal=solve(solver,frames,cfg['temporal_arms'],cfg['weighted_convergence'])
            collision=[];refined=[];delta=None
            for i,(frame,seed) in enumerate(zip(frames,q)):
                value,info=refine(solver,frame,seed,previous_correction=delta,**policy['collision_settings'])
                delta=value[7:]-seed[7:];refined.append(value);collision.append(info)
            original=np.asarray(refined);q=original.copy();ground_info=None;rate_info=None;rejections=[];review_flags=[]
            scene=scenes[sid]
            if scene['scene']['core_eligible']:
                geometry=replay(model,q)[2]
                if geometry['ground_penetration_p95_m']>post['ground_trigger_p95_m'] or geometry['ground_penetration_max_m']>post['ground_trigger_max_m']:
                    q,ground_info=correct(model,q,permitted_penetration=post['ground_target_penetration_m'])
                q[:,7:],rate_info=project(q[:,7:],model.jnt_range[1:,0],model.jnt_range[1:,1],limits,dt=.04,correction_curvature=post['rate_projection_correction_curvature'])
                if rate_info['correction_max_rad']>post['maximum_rate_projection_angle_change_rad']:rejections.append('RATE_CORRECTION_EXCEEDS_CONTENT_CHANGE_BUDGET')
            else:rejections.append('SCENE_NOT_YET_VALIDATED_FOR_TRAINING')
            before=target_errors(solver,frames,original,cfg);after=target_errors(solver,frames,q,cfg)
            change={'position_worst_increase_m':float((after[0]-before[0]).max()),'orientation_worst_increase_deg':float((after[1]-before[1]).max())}
            guard=policy['human_target_correction_review']
            if change['position_worst_increase_m']>guard['position_increase_above_m'] or change['orientation_worst_increase_deg']>guard['orientation_increase_above_deg']:
                review_flags.append('CORRECTION_CONTENT_REVIEW_REQUIRED')
            save_array(qp,q);save_array(bp,original);write_json(tp,{'source':mapping,'reference':refmap})
            write_json(sp,{'temporal':temporal,'collision':collision,'scene':scene,'ground_correction':ground_info,'rate_projection':rate_info,
                          'human_target_residual_change':change,'quality_rejection_reasons':rejections,'review_flags':review_flags})
            align=initial_alignment(q[0],reference[0]);turn=R.from_euler('z',align['yaw_rad']);shift=np.asarray(align['translation'])
            error=np.rad2deg(abs(q[:,7:]-reference[:,7:]))
            entry.update(status='DIAGNOSTIC_COMPLETE',retargeted=True,reference_loaded=True,frames=len(q),fixed_planar_alignment=align,
                hinge_mae_deg=float(error.mean()),per_joint_mae_deg=error.mean(axis=0).tolist(),
                root_rmse_aligned_m=float(np.sqrt(np.mean(np.sum((turn.apply(q[:,:3])+shift-reference[:,:3])**2,axis=1)))),
                root_rmse_raw_m=float(np.sqrt(np.mean(np.sum((q[:,:3]-reference[:,:3])**2,axis=1)))),
                root_orientation_mean_deg=float(np.rad2deg((R.from_quat(reference[:,3:7],scalar_first=True).inv()*turn*R.from_quat(q[:,3:7],scalar_first=True)).magnitude()).mean()),
                candidate_continuity=continuity(model,q,times),reference_continuity=continuity(model,reference,times),
                scene_core_candidate=scene['scene']['core_eligible'],scene_class=scene['scene']['scene_class'],
                quality_rejection_reasons=rejections,review_flags=review_flags,human_target_residual_change=change,
                **{k:sha(p) for p,k in bindings})
        except Exception as exc:entry.update(error=repr(exc),traceback=__import__('traceback').format_exc())
        entry.update(elapsed_sec=time.monotonic()-started,attempt=old.get('attempt',0)+1)
        results.append(entry);write_jsonl(out/'results.jsonl',results);print(sid,entry['status'],entry.get('error',''),flush=True)
    report={'status':'FROZEN_PANEL_EXECUTED','fixed_panel_size':expected,'actual_attempted':len(results),
            'retargeted':sum(r['retargeted'] for r in results),'reference_loaded':sum(r['reference_loaded'] for r in results),
            'error_count':sum(r['status']=='ERROR' for r in results),'run_config':design,'results':results}
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',report)
    if verification:
        diffs=[]
        for r in results:
            if r['status']!='DIAGNOSTIC_COMPLETE':continue
            mid=r['sample_id'].split(':',1)[1];a=np.load(out/(mid+'.npy'));b=np.load(ROOT/'work/bones_core_dev_v24'/(mid+'.npy'))
            diffs.append({'sample_id':r['sample_id'],'max_abs_difference':float(abs(a-b).max())})
        result={'status':'PASS' if len(diffs)==3 and max(r['max_abs_difference'] for r in diffs)<=1e-6 else 'FAIL',
                'policy_sha256':sha(PATH),'results':diffs,'source_report_sha256':sha(ROOT/'qa/aggregate'/f'{namespace}.json')}
        write_json(ROOT/'audit/frozen_v25_reproduction.json',result);print(result,flush=True)
    else:
        from .frozen_pair_qa import run as paired
        from .summarize_bones_qa import summarize
        from .frozen_target_residual_qa import run as targets
        from .frozen_relative_qa import run as relative
        from .frozen_joint_error_qa import run as joints
        paired(namespace,panel);summarize(namespace);targets(namespace);relative(namespace,panel);joints(namespace,panel)
        from .frozen_calibration_evaluation import evaluate
        evaluate(namespace,panel)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=['smoke','dev','accept'],default='accept')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--verification',action='store_true')
    a=p.parse_args();print(run(a.panel,a.resume,a.dry_run,a.verification)['status'] if not a.dry_run else run(a.panel,a.resume,True,a.verification))
