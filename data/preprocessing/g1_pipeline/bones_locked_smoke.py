"""Fixed paired smoke diagnostic on project G1. No downstream release gate."""
import argparse
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .common import ROOT, BONES, read_jsonl, write_json, write_jsonl, sha, digest
from .robot import read_csv, FORMAT, compile_target
from .resample import qpos as resample_qpos, time_grid
from .gmr_soma import parse, MAP
from .locked_target import prepare, create_retargeter

IDENTITY_MAP={key:key for key in MAP}

def run(max_clips=16, *, config_path=None, namespace='bones_locked_smoke_v1', panel='smoke', resume=False):
    from pathlib import Path
    if not namespace.replace('_','').isalnum(): raise ValueError('INVALID_DIAGNOSTIC_NAMESPACE')
    if panel not in ('smoke','dev'): raise ValueError('ACCEPTANCE_REQUIRES_FROZEN_ADAPTER_AND_QA_POLICY')
    xml,asset=prepare()
    base=ROOT/'deps/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json'
    if config_path is None:
        config=json.loads(base.read_text()); inverse={v:k for k,v in MAP.items()}
        config['human_root_name']='Hips'
        config['human_scale_table']={inverse[k]:v for k,v in config['human_scale_table'].items()}
        for table in ('ik_match_table1','ik_match_table2'):
            for entry in config[table].values(): entry[0]=inverse[entry[0]]
        config_path=ROOT/'configs/bones_locked_smoke_v1.json'; write_json(config_path,config)
    else:
        config_path=Path(config_path);config=json.loads(config_path.read_text())
        if namespace=='bones_locked_smoke_v1': raise ValueError('CUSTOM_CONFIG_REQUIRES_NEW_NAMESPACE')
    source_mapping={name:name for name in config['human_scale_table']}
    manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl'
    all_rows=read_jsonl(manifest);rows=all_rows[:max_clips]
    output=ROOT/'work'/namespace;output.mkdir(exist_ok=True)
    previous_run=json.loads((output/'run.json').read_text()) if (output/'run.json').exists() else None
    previous={r['sample_id']:r for r in read_jsonl(output/'results.jsonl')} if resume and (output/'results.jsonl').exists() else {}
    target,_=compile_target();results=[]
    run_config={'mode':'paired_development_only','panel':panel,'asset_hash':asset['asset_hash'],
        'manifest_sha256':sha(manifest),
        'ik_config_sha256':sha(config_path),'ik_baseline_sha256':sha(base),
        'source_adapter':'bvh_bones_soma; explicit SOMA names; development config NOT release validated',
        'source_rotation_basis':'C @ world_rotation; local joint axes preserved',
        'scale_strategy':'explicit config table; no actor height applied to uniform SOMA',
        'source_config_path':str(config_path),
        'fps':25,'resampling':'local SO(3) SLERP and local translation interpolation BEFORE FK',
        'workers':1,'max_clips':max_clips,'state_reset':'new GMR instance per clip',
        'ground_strategy':'none; do not remove height errors',
        'time_rule':'common physical support; fail unexplained native frame-count mismatch',
        'code_hash':digest({p.name:sha(p) for p in [__import__('pathlib').Path(__file__),ROOT/'g1_pipeline/locked_target.py',ROOT/'g1_pipeline/gmr_soma.py',ROOT/'g1_pipeline/resample.py',
            ROOT/'g1_pipeline/gmr_initialization_diagnostic.py',ROOT/'g1_pipeline/gmr_weighted_convergence.py']}),
        'weighted_convergence':config.get('weighted_convergence')}
    write_json(output/'run.json',run_config)
    for row in rows:
        result={'sample_id':row['sample_id'],'status':'ERROR','retargeted':False,'reference_loaded':False}
        try:
            source_dir='gmr_soma_smoke' if panel=='smoke' else 'bones_pairs_dev'
            bvh=ROOT/'work'/source_dir/(row['source_motion_id']+'.bvh')
            ref_path=BONES/row['source_paths']['g1']
            cached=previous.get(row['sample_id'],{})
            qp=output/(row['source_motion_id']+'.npy');tp=output/(row['source_motion_id']+'.time.json')
            if previous_run==run_config and cached.get('status')=='DIAGNOSTIC_COMPLETE' and qp.exists() and tp.exists():
                if cached.get('source_sha256')==sha(bvh) and cached.get('reference_sha256')==sha(ref_path) and cached.get('output_sha256')==sha(qp) and cached.get('time_sha256')==sha(tp):
                    saved=np.load(qp)
                    if saved.shape==(cached['frames'],36) and np.isfinite(saved).all():
                        results.append(cached);print(row['sample_id'],'RESUME_VERIFIED',flush=True);continue
            lines=bvh.read_text().splitlines(); mi=lines.index('MOTION')
            n=int(lines[mi+1].split(':')[1]);dt=float(lines[mi+2].split(':')[1])
            ref,rt,rfps=read_csv(ref_path,format_id=FORMAT,fps=row['original_fps'])
            if n!=len(ref): raise ValueError(f'UNVERIFIED_PAIR_FRAME_COUNTS:{n}!={len(ref)}')
            end=min((n-1)*dt,rt[-1]);times,mapping=time_grid(np.arange(n)*dt,end=np.nextafter(end,np.inf))
            ref,tref,refmap=resample_qpos(ref,rt,target.jnt_range[1:],end=np.nextafter(end,np.inf))
            if not np.allclose(tref,times,atol=1e-10,rtol=0): raise ValueError('PAIR_TIME_MISMATCH')
            result['reference_loaded']=True
            frames,_=parse(bvh,source_mapping,target_timestamps=times,rotation_basis='world_only')
            solver=create_retargeter(config_path,xml);q=[];errors=[]
            if config.get('initialize_root_from_human',False):
                from .locked_target import initialize_root_from_human
                initialize_root_from_human(solver,frames[0])
            if config.get('initialize_orientation_seed',False):
                from .gmr_initialization_diagnostic import orientation_seed
                orientation_seed(solver,frames[0])
            body_residuals={body:[] for body in config['ik_match_table1']}
            convergence_reports=[]
            for frame in frames:
                if config.get('weighted_convergence'):
                    from .gmr_weighted_convergence import retarget
                    pose,iteration_reports=retarget(solver,frame,config['weighted_convergence'])
                    q.append(pose);convergence_reports.extend(iteration_reports)
                else:
                    q.append(solver.retarget(frame,offset_to_ground=False).copy())
                errors.append([float(solver.error1()),float(solver.error2())])
                data=solver.configuration.data;mujoco.mj_forward(solver.model,data)
                for body,entry in config['ik_match_table1'].items():
                    pos,quat=solver.scaled_human_data[entry[0]];bid=solver.model.body(body).id
                    pe=float(np.linalg.norm(data.xpos[bid]-pos))
                    re=float(np.rad2deg((Rotation.from_matrix(data.xmat[bid].reshape(3,3)).inv()*Rotation.from_quat(quat,scalar_first=True)).magnitude()))
                    body_residuals[body].append([pe,re])
            q=np.asarray(q)
            if q.shape!=(len(times),36) or not np.isfinite(q).all(): raise ValueError('INVALID_GMR_OUTPUT')
            import io
            from .common import atomic_bytes
            buffer=io.BytesIO();np.save(buffer,q);atomic_bytes(qp,buffer.getvalue())
            write_json(output/(row['source_motion_id']+'.time.json'),{'source':mapping,'reference':refmap})
            result['retargeted']=True
            if convergence_reports:
                from collections import Counter
                result['convergence']={str(stage):{
                    'stop_reasons':dict(Counter(r['reason'] for r in convergence_reports if r['stage']==stage)),
                    'iterations_mean':float(np.mean([r['accepted_iterations'] for r in convergence_reports if r['stage']==stage])),
                    'final_cost_mean':float(np.mean([r['final_cost'] for r in convergence_reports if r['stage']==stage])),
                    'final_cost_max':float(max(r['final_cost'] for r in convergence_reports if r['stage']==stage)),
                    'objective_increases':sum(any(np.diff(r['cost_history'])>0) for r in convergence_reports if r['stage']==stage)
                    } for stage in (1,2)}
            result['body_target_residuals']={body:{
                'source':config['ik_match_table1'][body][0],
                'position_mean_m':float(np.mean(values,axis=0)[0]),'position_p95_m':float(np.percentile(values,95,axis=0)[0]),
                'orientation_mean_deg':float(np.mean(values,axis=0)[1]),'orientation_p95_deg':float(np.percentile(values,95,axis=0)[1]),
                'task1_position_orientation_weights':config['ik_match_table1'][body][1:3],
                'task2_position_orientation_weights':config['ik_match_table2'][body][1:3]
                } for body,values in body_residuals.items()}
            result.update(frames=len(q),finite=bool(np.isfinite(q).all()),
                unit_quaternion_max_error=float(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=1)-1))),
                hinge_limit_max_violation_rad=float(max(0,np.max(target.jnt_range[1:,0]-q[:,7:]),np.max(q[:,7:]-target.jnt_range[1:,1]))),
                source_sha256=sha(bvh),reference_sha256=sha(ref_path),source_fps=1/dt,reference_fps=rfps,
                output_sha256=sha(qp),time_sha256=sha(tp),
                source_end_sec=(n-1)*dt,reference_end_sec=float(rt[-1]),output_end_sec=float(times[-1]),
                root_rmse_raw_m=float(np.sqrt(np.mean(np.sum((q[:,:3]-ref[:,:3])**2,axis=1)))),
                hinge_mae_deg=float(np.mean(np.abs(np.rad2deg(q[:,7:]-ref[:,7:])))),
                per_joint_mae_deg=np.mean(np.abs(np.rad2deg(q[:,7:]-ref[:,7:])),axis=0).tolist(),
                ik_weighted_error_max=np.max(errors,axis=0).tolist())
            # One initial planar translation + heading transform, no height fit.
            rc=Rotation.from_quat(q[:,3:7],scalar_first=True);rr=Rotation.from_quat(ref[:,3:7],scalar_first=True)
            cdir=rc[0].apply([1,0,0]);rdir=rr[0].apply([1,0,0])
            yaw=np.arctan2(rdir[1],rdir[0])-np.arctan2(cdir[1],cdir[0]);align=Rotation.from_euler('z',yaw)
            translation=ref[0,:3]-align.apply(q[0,:3]);translation[2]=0
            result['fixed_planar_alignment']={'yaw_rad':float(yaw),'translation':translation.tolist()}
            result['root_rmse_aligned_m']=float(np.sqrt(np.mean(np.sum((align.apply(q[:,:3])+translation-ref[:,:3])**2,axis=1))))
            result['root_orientation_mean_deg']=float(np.rad2deg((rr.inv()*align*rc).magnitude()).mean())
            result['status']='DIAGNOSTIC_COMPLETE'
        except Exception as exc:
            result['error']=repr(exc)
            result['traceback']=__import__('traceback').format_exc()
        results.append(result)
        write_jsonl(output/'results.jsonl',results)
        print(result['sample_id'],result['status'],result.get('error',''),flush=True)
    report={'status':'DIAGNOSTIC_ONLY','actual_attempted':len(results),'fixed_panel_size':len(all_rows),
        'reference_loaded':sum(r['reference_loaded'] for r in results),
        'retargeted':sum(r['retargeted'] for r in results),'error_count':sum(r['status']=='ERROR' for r in results),
        'asset_equivalence':asset['status'],'release_gate':'NOT_EVALUATED',
        'reason':'source offsets/scale not frozen; required contact/kinematic/independent acceptance metrics incomplete',
        'run_config':run_config,'results':results}
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-clips',type=int,default=16)
    p.add_argument('--config');p.add_argument('--namespace',default='bones_locked_smoke_v1')
    p.add_argument('--panel',choices=['smoke','dev'],default='smoke');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not 1<=a.max_clips<=(16 if a.panel=='smoke' else 200): p.error('clip count exceeds fixed panel budget')
    r=run(a.max_clips,config_path=a.config,namespace=a.namespace,panel=a.panel,resume=a.resume); print({k:v for k,v in r.items() if k not in ('run_config','results')})
