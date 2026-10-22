"""Fixed panels: floor/rate corrections within source-evidenced flat-core scope.

Every input retains its paired measurement. Unvalidated scenes are explicitly
not training-eligible, and a pose-change guard can reject a proposed correction.
"""
import argparse
import io
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha,atomic_bytes
from .gates import require_pass
from .bones_scene_evidence import load as load_scenes
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample
from .ground_clearance import correct
from .bounded_hinge_trajectory import project,sonic_training_reference_limits
from .bones_continuity_qa import metrics as continuity


def run(panel='smoke'):
    require_pass('00_preflight')
    if panel not in ('smoke','dev'):raise ValueError('HELD_OUT_REQUIRES_FROZEN_CONFIGURATION')
    namespace=f'bones_core_{panel}_v24';out=ROOT/'work'/namespace
    if out.exists():raise ValueError('DEVELOPMENT_NAMESPACE_EXISTS')
    out.mkdir();base=f'bones_collision_{panel}_v19';base_root=ROOT/'work'/base
    runpath=ROOT/'qa/aggregate'/f'{base}.json';run=json.loads(runpath.read_text())
    manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl';meta={r['sample_id']:r for r in read_jsonl(manifest)}
    scene_rows=load_scenes(manifest);scenes={r['sample_id']:r for r in scene_rows}
    qapath=base_root/'pair_qa.json';qa={r['sample_id']:r for r in json.loads(qapath.read_text())['results']}
    expected=16 if panel=='smoke' else 200
    if len(meta)!=expected or len(run['results'])!=expected:raise ValueError('FIXED_PANEL_SIZE_MISMATCH')
    cfgpath=ROOT/'configs/bones_core_refinement_v24.json';cfg=json.loads(cfgpath.read_text())
    model,asset=compile_target();limits=sonic_training_reference_limits(model)
    modules=['bones_core_development','bones_scene_evidence','scene_policy','ground_clearance',
             'bounded_hinge_trajectory','bones_continuity_qa','bones_pair_qa','summarize_bones_qa','robot','resample']
    design={'panel':panel,'expected':expected,'configuration':cfg,'configuration_sha256':sha(cfgpath),
            'base_run_sha256':sha(runpath),'base_qa_sha256':sha(qapath),'manifest_sha256':sha(manifest),
            'modules':{n:sha(ROOT/'g1_pipeline'/f'{n}.py') for n in modules},
            'asset_hash':asset['asset_hash'],'source_metadata_sha256':scene_rows[0]['metadata_sha256'],
            'rate_limits_rad_s':limits.tolist(),'reference_used_in_corrections':False,'time_changed':False,
            'resources':{'workers':1,'cpu_threads':1,'gpu':False},'release_allowed':False,
            'unsupported_scenes_keep_pairwise_denominator':True}
    write_json(out/'run.json',design);results=[]
    for bound in run['results']:
        sid=bound['sample_id'];row=meta[sid];mid=row['source_motion_id'];scene=scenes[sid]
        qp=base_root/(mid+'.npy');tp=qp.with_suffix('.time.json')
        if sha(qp)!=bound['output_sha256'] or sha(tp)!=bound['time_sha256']:raise ValueError('BASE_HASH_CHANGED')
        q=np.load(qp);t=np.asarray(json.loads(tp.read_text())['source']['target_timestamps']);output=q.copy()
        info={'scene':scene,'ground_correction':None,'rate_projection':None,'quality_rejection_reasons':[]}
        if scene['scene']['core_eligible']:
            g=qa[sid]['candidate_geometry']
            if g['ground_penetration_p95_m']>cfg['ground_trigger_p95_m'] or g['ground_penetration_max_m']>cfg['ground_trigger_max_m']:
                output,info['ground_correction']=correct(model,output,permitted_penetration=cfg['ground_target_penetration_m'])
            output[:,7:],info['rate_projection']=project(output[:,7:],model.jnt_range[1:,0],model.jnt_range[1:,1],limits,
                dt=cfg['physical_dt_s'],correction_curvature=cfg['rate_projection_correction_curvature'])
            if info['rate_projection']['correction_max_rad']>cfg['maximum_rate_projection_angle_change_rad']:
                info['quality_rejection_reasons'].append('RATE_CORRECTION_EXCEEDS_CONTENT_CHANGE_BUDGET')
        else:info['quality_rejection_reasons'].append('SCENE_NOT_YET_VALIDATED_FOR_TRAINING')
        if not np.array_equal(output[:,[0,1,3,4,5,6]],q[:,[0,1,3,4,5,6]]):raise ValueError('FORBIDDEN_ROOT_ORIENTATION_XY_CHANGE')
        op=out/(mid+'.npy');buf=io.BytesIO();np.save(buf,output);atomic_bytes(op,buf.getvalue());atomic_bytes(out/(mid+'.time.json'),tp.read_bytes())
        write_json(out/(mid+'.solver.json'),info)
        reference,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps']);reference,_,_=resample(reference,rt,model.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
        error=np.rad2deg(abs(output[:,7:]-reference[:,7:]));align=bound['fixed_planar_alignment'];turn=R.from_euler('z',align['yaw_rad']);shift=np.asarray(align['translation'])
        if shift[2]!=0:raise ValueError('ROOT_HEIGHT_ALIGNMENT_FORBIDDEN')
        entry={**bound,'output_sha256':sha(op),'solver_sha256':sha(out/(mid+'.solver.json')),
               'base_output_sha256':sha(qp),'hinge_mae_deg':float(error.mean()),'per_joint_mae_deg':error.mean(axis=0).tolist(),
               'root_rmse_aligned_m':float(np.sqrt(np.mean(np.sum((turn.apply(output[:,:3])+shift-reference[:,:3])**2,axis=1)))),
               'root_rmse_raw_m':float(np.sqrt(np.mean(np.sum((output[:,:3]-reference[:,:3])**2,axis=1)))),
               'candidate_continuity':continuity(model,output,t),'scene_class':scene['scene']['scene_class'],
               'scene_core_candidate':scene['scene']['core_eligible'],'quality_rejection_reasons':info['quality_rejection_reasons'],
               'ground_lift_max_m':info['ground_correction']['lift_max_m'] if info['ground_correction'] else 0.,
               'rate_correction_max_rad':info['rate_projection']['correction_max_rad'] if info['rate_projection'] else 0.}
        results.append(entry);print(sid,entry['ground_lift_max_m'],entry['rate_correction_max_rad'],flush=True)
    result={'status':'DEVELOPMENT_ONLY','fixed_panel_size':expected,'actual_attempted':expected,'retargeted':expected,
            'reference_loaded':expected,'error_count':0,'run_config':design,'results':results}
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',result)
    from .bones_pair_qa import run as pair_qa
    from .summarize_bones_qa import summarize
    pair_qa(namespace,panel);summarize(namespace)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=['smoke','dev'],default='smoke');run(p.parse_args().panel)
