"""Continue GMR from time-regularized arm seeds on the fixed dev subset."""
import io
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha,atomic_bytes,digest
from .locked_target import create_retargeter
from .gmr_soma import parse
from .gmr_arm_trajectory_seed import ArmChain
from .gmr_weighted_convergence import converge,cost
from .initial_planar_alignment import initial_alignment
from .robot import read_csv,FORMAT
from .resample import qpos as resample
from .bones_continuity_qa import metrics as continuity


def run():
    namespace='bones_arm_gmr_refinement_v14';output=ROOT/'work'/namespace
    if output.exists():raise ValueError('DIAGNOSTIC_NAMESPACE_ALREADY_EXISTS')
    output.mkdir()
    cfgpath=ROOT/'configs/bones_soma_weighted_convergence_v8.json';cfg=json.loads(cfgpath.read_text())
    selection=ROOT/'audit/bones_arm_branch_path_v13.json';selected=json.loads(selection.read_text())['results']
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';rows={r['sample_id']:r for r in read_jsonl(manifest)}
    design={'panel':'dev','selection_sha256':sha(selection),'manifest_sha256':sha(manifest),'config_sha256':sha(cfgpath),
            'method':'GMR stage2 plus arm-only PostureTask centered on human-only continuous full-clip seed',
            'arm_posture_cost':20.,'max_iterations':50,'relative_tolerance':1e-6,
            'initial_alignment':'relative_SO3_projected_yaw_v2','status':'DEVELOPMENT_ONLY',
            'code_hashes':{p.name:sha(p) for p in [ROOT/'g1_pipeline/diagnose_arm_gmr_refinement.py',ROOT/'g1_pipeline/gmr_weighted_convergence.py']}}
    write_json(output/'run.json',design);results=[]
    for item in selected:
        row=rows[item['sample_id']];mid=row['source_motion_id'];seedpath=ROOT/'work/bones_arm_branch_path_v13'/(mid+'.npy')
        seed=np.load(seedpath);tp=ROOT/'work/bones_soma_rootweight_dev_v6'/(mid+'.time.json')
        times=np.array(json.loads(tp.read_text())['source']['target_timestamps'])
        frames,_=parse(ROOT/'work/bones_pairs_dev'/(mid+'.bvh'),{n:n for n in cfg['human_scale_table']},target_timestamps=times,rotation_basis='world_only')
        solver=create_retargeter(cfgpath,ROOT/'assets/project_g1_v1/g1.xml')
        import mink
        weights=np.zeros(solver.model.nv)
        for side in ('left','right'):weights[solver.model.jnt_dofadr[ArmChain(solver.model,side).ids]]=20.
        posture=mink.PostureTask(solver.model,cost=weights)
        q=[];objectives=[]
        for pose,frame in zip(seed,frames):
            solver.update_targets(frame,offset_to_ground=False);solver.configuration.update(pose);posture.set_target(pose)
            converge(solver,solver.tasks2+[posture]);q.append(solver.configuration.q.copy())
            objectives.append(cost(solver.configuration,solver.tasks2))
        q=np.array(q);path=output/(mid+'.npy');buf=io.BytesIO();np.save(buf,q);atomic_bytes(path,buf.getvalue())
        atomic_bytes(output/(mid+'.time.json'),tp.read_bytes())
        ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
        ref,tt,_=resample(ref,rt,solver.model.jnt_range[1:],end=np.nextafter(times[-1],np.inf))
        if q.shape!=ref.shape or not np.allclose(tt,times,atol=1e-10,rtol=0):raise ValueError('TIME_MISMATCH')
        align=initial_alignment(q[0],ref[0]);rotation=R.from_euler('z',align['yaw_rad']);translation=np.array(align['translation'])
        entry={'sample_id':row['sample_id'],'status':'DIAGNOSTIC_COMPLETE','frames':len(q),
               'retargeted':True,'reference_loaded':True,'seed_sha256':sha(seedpath),'output_sha256':sha(path),
               'hinge_mae_deg':float(np.rad2deg(abs(q[:,7:]-ref[:,7:])).mean()),
               'fixed_planar_alignment':align,'candidate_continuity':continuity(solver.model,q,times),
               'root_rmse_aligned_m':float(np.sqrt(np.mean(np.sum((rotation.apply(q[:,:3])+translation-ref[:,:3])**2,axis=1)))),
               'root_orientation_mean_deg':float(np.rad2deg((R.from_quat(ref[:,3:7],scalar_first=True).inv()*rotation*R.from_quat(q[:,3:7],scalar_first=True)).magnitude()).mean()),
               'human_task2_cost_mean':float(np.mean(objectives))}
        results.append(entry)
        write_json(ROOT/'qa/aggregate'/f'{namespace}.json',{'status':'DEVELOPMENT_SUBSET_ONLY','fixed_panel_size':200,'actual_attempted':len(results),'run_config':design,'results':results})
        print(row['sample_id'],entry['hinge_mae_deg'],max(entry['candidate_continuity']['hinge_speed_max_rad_s']),flush=True)
    from .bones_pair_qa import run as paired_qa
    from .summarize_bones_qa import summarize
    paired_qa(namespace,'dev',[r['sample_id'] for r in results]);summarize(namespace)
    return results


if __name__=='__main__':run()
