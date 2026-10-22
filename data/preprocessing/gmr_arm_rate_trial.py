"""Two development shoulder branch failures: fit human targets with rate penalties.

Experimental, not a production runner. All original frames and non-arm poses
are preserved; the previous rejected angle-only projection supplies a seed.
"""
import io
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha,atomic_bytes,BONES
from g1_pipeline.gates import require_pass
from g1_pipeline.locked_target import create_retargeter
from g1_pipeline.gmr_soma import parse
from g1_pipeline.gmr_arm_trajectory_seed import ArmChain,se3_error
from g1_pipeline.bounded_hinge_trajectory import project,sonic_training_reference_limits
from g1_pipeline.bones_continuity_qa import metrics as continuity
from g1_pipeline.robot import read_csv,FORMAT
from g1_pipeline.resample import qpos as resample


def optimize(chain,initial,pp,pr,tp,tr,weights,limits):
    n=len(initial);rows=n*18+(n-1)*14+(n-2)*7
    structure=lil_matrix((rows,n*7),dtype=np.int8)
    for i in range(n):structure[i*18:(i+1)*18,i*7:(i+1)*7]=1
    offset=n*18
    for repeat in range(2):
        for i in range(n-1):
            for j in range(7):structure[offset+i*7+j,[i*7+j,(i+1)*7+j]]=1
        offset+=(n-1)*7
    for i in range(n-2):
        for j in range(7):structure[offset+i*7+j,[i*7+j,(i+1)*7+j,(i+2)*7+j]]=1
    def residual(x):
        a=x.reshape(n,7);p,r=chain.fk(a,pp,pr);v=np.diff(a,axis=0)/.04
        return np.concatenate([(se3_error(p,r,tp,tr)*weights).ravel(),(.8*v).ravel(),
                               (100*np.maximum(abs(v)-limits,0)).ravel(),
                               (.02*np.diff(a,n=2,axis=0)/.04**2).ravel()])
    lo=np.tile(chain.bounds[:,0]+1e-9,n);hi=np.tile(chain.bounds[:,1]-1e-9,n)
    initial=np.clip(initial.ravel(),lo,hi);before=residual(initial)
    fit=least_squares(residual,initial,bounds=(lo,hi),jac_sparsity=structure.tocsr(),max_nfev=200,
                      ftol=1e-6,xtol=1e-7,gtol=1e-5)
    result,projection=project(fit.x.reshape(n,7),chain.bounds[:,0],chain.bounds[:,1],limits)
    return result,{'initial_cost':float(before@before),'final_cost':float(fit.fun@fit.fun),
                   'nfev':fit.nfev,'success':bool(fit.success),'optimality':float(fit.optimality),
                   'final_constraint_projection':projection}


def run():
    require_pass('00_preflight')
    namespace='bones_arm_rate_human_v21';out=ROOT/'work'/namespace
    if out.exists():raise ValueError('DIAGNOSTIC_NAMESPACE_EXISTS')
    out.mkdir();selected=['bones:cartwheel_R_005__A238','bones:cartwheelin_001__A487']
    meta={r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')}
    baseline_path=ROOT/'qa/aggregate/bones_rate_projection_v20.json'
    base={r['sample_id']:r for r in json.loads(baseline_path.read_text())['results']}
    cfgpath=ROOT/'configs/bones_temporal_arms_v15.json';cfg=json.loads(cfgpath.read_text())
    design={'panel':'dev','scope':'two known development shoulder branch failures; human-task rate-constrained refinement',
            'reference_input_to_optimizer':False,'rate_penalty_weight':100.,'velocity_weight':.8,'acceleration_weight':.02,
            'max_nfev':200,'root_torso_legs_fixed':True,'sample_ids':selected,'held_out_read':False,
            'hashes':{str(p):sha(p) for p in [baseline_path,cfgpath,ROOT/'gmr_arm_rate_trial.py',
                         ROOT/'g1_pipeline/gmr_arm_trajectory_seed.py',ROOT/'g1_pipeline/bounded_hinge_trajectory.py']},
            'resources':{'workers':1,'cpu_threads':1,'gpu':False},'production_adoption':False}
    write_json(out/'run.json',design);results=[]
    for sid in selected:
        row=meta[sid];mid=row['source_motion_id'];seedpath=ROOT/'work/bones_rate_projection_v20'/(mid+'.npy')
        timepath=seedpath.with_suffix('.time.json');q=np.load(seedpath);t=np.asarray(json.loads(timepath.read_text())['source']['target_timestamps'])
        frames,_=parse(ROOT/'work/bones_pairs_dev'/(mid+'.bvh'),{n:n for n in cfg['human_scale_table']},target_timestamps=t,rotation_basis='world_only')
        solver=create_retargeter(cfgpath,ROOT/'assets/project_g1_v1/g1.xml');m=solver.model;d=mujoco.MjData(m)
        limits=sonic_training_reference_limits(m);torso=m.body('torso_link').id;pp=[];pr=[];targets=[];reports=[]
        for pose,frame in zip(q,frames):
            d.qpos[:]=pose;mujoco.mj_kinematics(m,d);pp.append(d.xpos[torso].copy());pr.append(d.xmat[torso].reshape(3,3).copy())
            solver.update_targets(frame,offset_to_ground=False)
            targets.append({k:(np.asarray(v[0]).copy(),np.asarray(v[1]).copy()) for k,v in solver.scaled_human_data.items()})
        pp=np.array(pp);pr=np.array(pr)
        for side in ('left','right'):
            chain=ArmChain(m,side);names=[cfg['ik_match_table2'][f'{side}_{b}_link'][0] for b in ('shoulder_yaw','elbow','wrist_yaw')]
            tp=np.array([[frame[n][0] for n in names] for frame in targets]);quat=np.array([[frame[n][1] for n in names] for frame in targets])
            tr=R.from_quat(quat.reshape(-1,4),scalar_first=True).as_matrix().reshape(len(q),3,3,3)
            weights=np.array([solver.human_body_to_task2[n].cost for n in names])
            q[:,chain.adr],info=optimize(chain,q[:,chain.adr],pp,pr,tp,tr,weights,limits[chain.adr-7]);reports.append({'side':side,**info})
            print(sid,side,info,flush=True)
        path=out/(mid+'.npy');buf=io.BytesIO();np.save(buf,q);atomic_bytes(path,buf.getvalue());atomic_bytes(out/(mid+'.time.json'),timepath.read_bytes())
        write_json(out/(mid+'.solver.json'),reports)
        ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps']);ref,_,_=resample(ref,rt,m.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
        error=np.rad2deg(abs(q[:,7:]-ref[:,7:]));result={**base[sid],'output_sha256':sha(path),'solver_sha256':sha(out/(mid+'.solver.json')),
          'hinge_mae_deg':float(error.mean()),'per_joint_mae_deg':error.mean(axis=0).tolist(),'candidate_continuity':continuity(m,q,t)}
        result.pop('rate_projection',None);results.append(result)
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',{'status':'DEVELOPMENT_SUBSET','fixed_panel_size':200,'actual_attempted':2,'run_config':design,'results':results})
    from g1_pipeline.bones_pair_qa import run as pair_qa
    from g1_pipeline.summarize_bones_qa import summarize
    from g1_pipeline.gmr_target_residual_qa import run as target_qa
    pair_qa(namespace,'dev',sample_ids=selected);summarize(namespace);target_qa(namespace)


if __name__=='__main__':run()
