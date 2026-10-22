"""Fixed BONES development runner for temporal arm GMR; held-out remains locked."""
import argparse
import io
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,write_jsonl,sha,digest,atomic_bytes
from .locked_target import prepare,create_retargeter
from .robot import read_csv,FORMAT,compile_target
from .resample import qpos as resample,time_grid
from .gmr_soma import parse
from .gmr_temporal_arms import solve
from .initial_planar_alignment import initial_alignment
from .bones_continuity_qa import metrics as continuity


def run(panel='smoke',resume=False):
    if panel not in ('smoke','dev'):raise ValueError('HELD_OUT_REQUIRES_FROZEN_CONFIGURATION')
    from .gates import require_pass
    require_pass('00_preflight')
    namespace=f'bones_temporal_arms_{panel}_v15';out=ROOT/'work'/namespace
    manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl';rows=read_jsonl(manifest)
    if len(rows)!=(16 if panel=='smoke' else 200):raise ValueError('FIXED_PANEL_SIZE_MISMATCH')
    cfgpath=ROOT/'configs/bones_temporal_arms_v15.json';cfg=json.loads(cfgpath.read_text())
    xml,asset=prepare();model,_=compile_target()
    modules=['bones_temporal_development','gmr_temporal_arms','gmr_arm_seed','gmr_arm_trajectory_seed','gmr_arm_branch_path',
             'gmr_weighted_convergence','gmr_soma','locked_target','resample','initial_planar_alignment']
    run_config={'panel':panel,'manifest_sha256':sha(manifest),'config_sha256':sha(cfgpath),'asset_hash':asset['asset_hash'],
                'modules':{n:sha(ROOT/'g1_pipeline'/f'{n}.py') for n in modules},'namespace':namespace,
                'fps':25,'workers':1,'gpu':False,'full_clips':True,'reference_input_to_solver':False,
                'configuration':cfg,'expected':len(rows),'estimated_output_bytes':sum(r['original_frames'] for r in rows)*36*8}
    old=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else None
    if old is not None and old!=run_config:raise ValueError('NAMESPACE_FINGERPRINT_MISMATCH_USE_NEW_NAMESPACE')
    if old is not None and not resume:raise ValueError('EXISTING_NAMESPACE_REQUIRES_RESUME')
    out.mkdir(exist_ok=True);write_json(out/'run.json',run_config)
    previous={r['sample_id']:r for r in read_jsonl(out/'results.jsonl')} if (out/'results.jsonl').exists() else {}
    results=[]
    for row in rows:
        mid=row['source_motion_id'];started=time.monotonic();path=out/(mid+'.npy');tp=out/(mid+'.time.json');dp=out/(mid+'.solver.json')
        bvh=ROOT/'work'/('gmr_soma_smoke' if panel=='smoke' else 'bones_pairs_dev')/(mid+'.bvh');rp=BONES/row['source_paths']['g1']
        entry={'sample_id':row['sample_id'],'status':'ERROR','retargeted':False,'reference_loaded':False}
        try:
            cached=previous.get(row['sample_id'],{})
            if resume and cached.get('status')=='DIAGNOSTIC_COMPLETE' and all(p.exists() and sha(p)==cached.get(k) for p,k in [(path,'output_sha256'),(tp,'time_sha256'),(dp,'solver_sha256'),(bvh,'source_sha256'),(rp,'reference_sha256')]):
                saved=np.load(path)
                if saved.shape==(cached['frames'],36) and np.isfinite(saved).all():
                    results.append(cached);continue
            lines=bvh.read_text().splitlines();ix=lines.index('MOTION');n=int(lines[ix+1].split(':')[1]);dt=float(lines[ix+2].split(':')[1])
            reference,rt,_=read_csv(rp,format_id=FORMAT,fps=row['original_fps'])
            if n!=len(reference):raise ValueError('UNVERIFIED_PAIR_FRAME_COUNTS')
            end=np.nextafter(min((n-1)*dt,rt[-1]),np.inf);times,mapping=time_grid(np.arange(n)*dt,end=end)
            reference,tt,refmap=resample(reference,rt,model.jnt_range[1:],end=end)
            if not np.allclose(times,tt,atol=1e-10,rtol=0):raise ValueError('PAIR_TIME_MISMATCH')
            entry['reference_loaded']=True
            frames,_=parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=times,rotation_basis='world_only')
            solver=create_retargeter(cfgpath,xml);q,diagnostic=solve(solver,frames,cfg['temporal_arms'],cfg['weighted_convergence'])
            if q.shape!=(len(times),36):raise ValueError('INVALID_OUTPUT_SHAPE')
            buf=io.BytesIO();np.save(buf,q);atomic_bytes(path,buf.getvalue());write_json(tp,{'source':mapping,'reference':refmap});write_json(dp,diagnostic)
            align=initial_alignment(q[0],reference[0]);turn=R.from_euler('z',align['yaw_rad']);shift=np.array(align['translation'])
            entry.update(status='DIAGNOSTIC_COMPLETE',retargeted=True,frames=len(q),source_sha256=sha(bvh),reference_sha256=sha(rp),
                         output_sha256=sha(path),time_sha256=sha(tp),solver_sha256=sha(dp),fixed_planar_alignment=align,
                         hinge_mae_deg=float(np.rad2deg(abs(q[:,7:]-reference[:,7:])).mean()),
                         per_joint_mae_deg=np.rad2deg(abs(q[:,7:]-reference[:,7:])).mean(axis=0).tolist(),
                         root_rmse_aligned_m=float(np.sqrt(np.mean(np.sum((turn.apply(q[:,:3])+shift-reference[:,:3])**2,axis=1)))),
                         root_rmse_raw_m=float(np.sqrt(np.mean(np.sum((q[:,:3]-reference[:,:3])**2,axis=1)))),
                         root_orientation_mean_deg=float(np.rad2deg((R.from_quat(reference[:,3:7],scalar_first=True).inv()*turn*R.from_quat(q[:,3:7],scalar_first=True)).magnitude()).mean()),
                         candidate_continuity=continuity(model,q,times),reference_continuity=continuity(model,reference,times))
        except Exception as exc:
            entry['error']=repr(exc);entry['traceback']=__import__('traceback').format_exc()
        entry['elapsed_sec']=time.monotonic()-started;results.append(entry);write_jsonl(out/'results.jsonl',results)
        print(entry['sample_id'],entry['status'],entry.get('hinge_mae_deg'),entry.get('error',''),flush=True)
    report={'status':'DEVELOPMENT_ONLY','fixed_panel_size':len(rows),'actual_attempted':len(results),
            'retargeted':sum(r['retargeted'] for r in results),'reference_loaded':sum(r['reference_loaded'] for r in results),
            'error_count':sum(r['status']=='ERROR' for r in results),'run_config':run_config,'results':results}
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',report)
    from .bones_pair_qa import run as pair_qa
    from .summarize_bones_qa import summarize
    pair_qa(namespace,panel);summarize(namespace)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=['smoke','dev'],default='smoke');p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.panel,a.resume)
