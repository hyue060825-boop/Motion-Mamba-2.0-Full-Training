"""Full seven-clip development trial of joint ground/contact refinement."""
import io
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT,BONES,read_jsonl,write_json,sha,atomic_bytes
from g1_pipeline.gates import require_pass
from g1_pipeline.gmr_ground_contact_refinement import refine
from g1_pipeline.locked_target import create_retargeter
from g1_pipeline.gmr_soma import parse
from g1_pipeline.robot import read_csv,FORMAT
from g1_pipeline.resample import qpos as resample
from g1_pipeline.bones_continuity_qa import metrics as continuity


def run():
    require_pass('00_preflight');namespace='bones_ground_contact_v23';out=ROOT/'work'/namespace
    if out.exists():raise ValueError('DIAGNOSTIC_NAMESPACE_EXISTS')
    out.mkdir();prior=ROOT/'work/bones_ground_clearance_v22/run.json';selected=json.loads(prior.read_text())['sample_ids']
    sourcepath=ROOT/'qa/aggregate/bones_collision_dev_v19.json';source={r['sample_id']:r for r in json.loads(sourcepath.read_text())['results']}
    metadata={r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')}
    config=ROOT/'configs/bones_temporal_arms_v15.json';cfg=json.loads(config.read_text())
    design={'panel':'dev','selection':'same seven v19 core ground failures as v22; all frames','sample_ids':selected,
            'root_z_max_adjustment_m':.15,'ground_penalty_min_distance_m':-.015,'self_penalty_min_distance_m':-.002,
            'penalty_weight':2000.,'root_z_posture_weight':75.,'hinge_posture_weight':5.,
            'root_z_correction_velocity_weight':1.5,'hinge_correction_velocity_weight':.4,
            'max_nfev':100,'reference_input':False,'root_height_error_not_aligned_away':True,
            'hashes':{str(p):sha(p) for p in [prior,sourcepath,config,ROOT/'g1_pipeline/gmr_ground_contact_refinement.py',
                      ROOT/'g1_pipeline/gmr_collision_refinement.py',ROOT/'scripts/diagnose_bones_ground_contact.py']},
            'resources':{'workers':1,'cpu_threads':1,'gpu':False},'production_adoption':False}
    write_json(out/'run.json',design);results=[]
    for sid in selected:
        row=metadata[sid];mid=row['source_motion_id'];bound=source[sid]
        qp=ROOT/'work/bones_collision_dev_v19'/(mid+'.npy');tp=qp.with_suffix('.time.json')
        if sha(qp)!=bound['output_sha256'] or sha(tp)!=bound['time_sha256']:raise ValueError('BASE_HASH_CHANGED')
        q=np.load(qp);t=np.asarray(json.loads(tp.read_text())['source']['target_timestamps'])
        bvh=ROOT/'work/bones_pairs_dev'/(mid+'.bvh')
        if sha(bvh)!=bound['source_sha256']:raise ValueError('SOURCE_CHANGED')
        frames,_=parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=t,rotation_basis='world_only')
        solver=create_retargeter(config,ROOT/'assets/project_g1_v1/g1.xml');poses=[];infos=[];previous=None
        for i,(frame,seed) in enumerate(zip(frames,q)):
            pose,info=refine(solver,frame,seed,previous);poses.append(pose);infos.append(info)
            previous=pose[np.r_[2,np.arange(7,36)]]-seed[np.r_[2,np.arange(7,36)]]
            if i%50==0:print(sid,i,'/',len(q),info['ground_excess_max_m'],flush=True)
        output=np.asarray(poses)
        if not np.array_equal(output[:,[0,1,3,4,5,6]],q[:,[0,1,3,4,5,6]]):raise ValueError('FORBIDDEN_ROOT_CHANGE')
        op=out/(mid+'.npy');buf=io.BytesIO();np.save(buf,output);atomic_bytes(op,buf.getvalue());atomic_bytes(out/(mid+'.time.json'),tp.read_bytes())
        write_json(out/(mid+'.solver.json'),{'frames':infos})
        ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps']);ref,_,_=resample(ref,rt,solver.model.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
        error=np.rad2deg(abs(output[:,7:]-ref[:,7:]));align=bound['fixed_planar_alignment'];turn=R.from_euler('z',align['yaw_rad']);shift=np.array(align['translation'])
        if shift[2]!=0:raise ValueError('ROOT_HEIGHT_ALIGNMENT_FORBIDDEN')
        result={**bound,'output_sha256':sha(op),'solver_sha256':sha(out/(mid+'.solver.json')),
                'hinge_mae_deg':float(error.mean()),'per_joint_mae_deg':error.mean(axis=0).tolist(),
                'root_rmse_aligned_m':float(np.sqrt(np.mean(np.sum((turn.apply(output[:,:3])+shift-ref[:,:3])**2,axis=1)))),
                'root_rmse_raw_m':float(np.sqrt(np.mean(np.sum((output[:,:3]-ref[:,:3])**2,axis=1)))),
                'candidate_continuity':continuity(solver.model,output,t),'root_z_correction_max_m':float(abs(output[:,2]-q[:,2]).max()),
                'optimizer_budget_frames':sum(not x['optimizer_success'] for x in infos)}
        results.append(result);write_json(ROOT/'qa/aggregate'/f'{namespace}.json',{'status':'DEVELOPMENT_SUBSET','fixed_panel_size':200,'actual_attempted':len(results),'run_config':design,'results':results})
    from g1_pipeline.bones_pair_qa import run as pair_qa
    from g1_pipeline.summarize_bones_qa import summarize
    pair_qa(namespace,'dev',sample_ids=selected);summarize(namespace)


if __name__=='__main__':run()
