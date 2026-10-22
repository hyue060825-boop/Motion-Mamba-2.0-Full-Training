"""Every-frame unweighted human-target position/orientation evidence for BONES."""
import argparse
import io
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from .common import ROOT,read_jsonl,write_json,sha,atomic_bytes
from .locked_target import create_retargeter
from .gmr_soma import parse


def run(namespace):
    if not namespace.replace('_','').isalnum():raise ValueError('INVALID_NAMESPACE')
    runpath=ROOT/'qa/aggregate'/f'{namespace}.json';report=json.loads(runpath.read_text())
    panel=report['run_config']['panel']
    if panel not in ('smoke','dev','accept'):raise ValueError('INVALID_PANEL')
    if panel=='accept':
        from .calibration_protocol import require_frozen
        require_frozen()
    manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl';metadata={r['sample_id']:r for r in read_jsonl(manifest)}
    config=ROOT/'configs/bones_temporal_arms_v15.json';cfg=json.loads(config.read_text());rows=[]
    output=ROOT/'qa/per_clip'/namespace/'human_targets';output.mkdir(parents=True,exist_ok=True)
    bodies=list(cfg['ik_match_table2']);body_metrics={b:[] for b in bodies}
    for item in report['results']:
        entry={'sample_id':item['sample_id'],'status':'ERROR'}
        try:
            row=metadata[item['sample_id']];mid=row['source_motion_id'];qpath=ROOT/'work'/namespace/(mid+'.npy')
            tp=qpath.with_suffix('.time.json');t=np.asarray(json.loads(tp.read_text())['source']['target_timestamps'])
            if sha(qpath)!=item['output_sha256'] or sha(tp)!=item['time_sha256']:raise ValueError('SEALED_OUTPUT_HASH_CHANGED')
            q=np.load(qpath);bvh=ROOT/'work'/('gmr_soma_smoke' if panel=='smoke' else 'bones_pairs_'+panel)/(mid+'.bvh')
            if sha(bvh)!=item['source_sha256']:raise ValueError('SOURCE_HASH_CHANGED')
            frames,_=parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=t,rotation_basis='world_only')
            if len(q)!=len(frames):raise ValueError('TARGET_TIME_LENGTH_MISMATCH')
            solver=create_retargeter(config,ROOT/'assets/project_g1_v1/g1.xml');m=solver.model
            ids=[m.body(b).id for b in bodies];human_names=[cfg['ik_match_table2'][b][0] for b in bodies]
            pe=[];oe=[]
            for pose,frame in zip(q,frames):
                solver.update_targets(frame,offset_to_ground=False);solver.configuration.update(pose);d=solver.configuration.data
                targets=solver.scaled_human_data
                target_p=np.array([targets[n][0] for n in human_names]);target_q=np.array([targets[n][1] for n in human_names])
                pe.append(np.linalg.norm(d.xpos[ids]-target_p,axis=1))
                oe.append(np.rad2deg((R.from_matrix(d.xmat[ids].reshape(-1,3,3)).inv()*R.from_quat(target_q,scalar_first=True)).magnitude()))
            pe=np.asarray(pe);oe=np.asarray(oe);path=output/(mid+'.npz')
            buf=io.BytesIO();np.savez_compressed(buf,timestamps=t,position_error_m=pe,orientation_error_deg=oe);atomic_bytes(path,buf.getvalue())
            metrics={}
            for j,body in enumerate(bodies):
                metrics[body]={'position_mean_m':float(pe[:,j].mean()),'position_p95_m':float(np.percentile(pe[:,j],95)),
                    'position_max_m':float(pe[:,j].max()),'orientation_mean_deg':float(oe[:,j].mean()),
                    'orientation_p95_deg':float(np.percentile(oe[:,j],95)),'orientation_max_deg':float(oe[:,j].max()),
                    'orientation_max_frame':int(oe[:,j].argmax()),'position_max_frame':int(pe[:,j].argmax())}
                body_metrics[body].append((metrics[body],len(q)))
            entry.update(status='MEASURED',frames=len(q),body_metrics=metrics,per_frame_path=str(path),
                         per_frame_sha256=sha(path),qpos_sha256=sha(qpath),time_sha256=sha(tp),source_sha256=sha(bvh))
        except Exception as exc:entry['error']=repr(exc)
        rows.append(entry);print(entry['sample_id'],entry['status'],entry.get('error',''),flush=True)
    aggregate={}
    for body,values in body_metrics.items():
        aggregate[body]={}
        for metric in ('position_mean_m','orientation_mean_deg','position_max_m','orientation_max_deg'):
            v=np.array([r[metric] for r,n in values]);weights=np.array([n for r,n in values])
            aggregate[body][metric]={'clip_macro_mean':float(v.mean()),'across_clip_p95':float(np.percentile(v,95)),
                                    'clip_max':float(v.max()),'frame_weighted_mean_of_clip_metric':float(np.average(v,weights=weights))} if len(v) else None
    result={'status':'HUMAN_TARGET_RESIDUAL_MEASUREMENTS','namespace':namespace,'expected_panel':report['fixed_panel_size'],
            'selected':len(report['results']),'measured':sum(r['status']=='MEASURED' for r in rows),'results':rows,
            'body_order':bodies,'aggregate':aggregate,'no_reference_robot_in_metric':True,
            'hashes':{str(p):sha(p) for p in [runpath,manifest,config,ROOT/'g1_pipeline/frozen_target_residual_qa.py']},
            'definition':'Euclidean body origin difference and SO3 geodesic against scaled offset GMR human targets; world coordinates; no ground or per-frame fit'}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_human_targets.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--namespace',required=True);run(p.parse_args().namespace)
