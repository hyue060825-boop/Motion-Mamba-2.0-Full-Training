"""Evaluate full-time arm warm starts on seven fixed development failures."""
import io
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha,atomic_bytes
from .locked_target import create_retargeter
from .gmr_soma import parse
from .gmr_arm_trajectory_seed import ArmChain,optimize
from .bones_continuity_qa import metrics
from .robot import read_csv,FORMAT
from .resample import qpos as resample


def run(branches=False,strong=False):
    cfgpath=ROOT/'configs/bones_soma_weighted_convergence_v8.json';cfg=json.loads(cfgpath.read_text())
    selection=ROOT/'audit/bones_arm_sequence_v10.json';selected=json.loads(selection.read_text())['results']
    rows={r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')}
    namespace='bones_arm_branch_path_v13' if strong else ('bones_arm_branch_path_v12' if branches else 'bones_arm_trajectory_seed_v11')
    if strong and not branches:raise ValueError('STRONG_REQUIRES_BRANCH_PATH')
    velocity_weight,acceleration_weight=(.8,.02) if strong else (.2,.004)
    output=ROOT/'work'/namespace
    if output.exists():raise ValueError('DIAGNOSTIC_NAMESPACE_ALREADY_EXISTS')
    output.mkdir()
    design={'status':'DEVELOPMENT_ARM_WARM_START_ONLY','selection_sha256':sha(selection),
            'config_sha256':sha(cfgpath),'velocity_weight':velocity_weight,'acceleration_weight':acceleration_weight,
            'dt':.04,'max_nfev':120,'expected':len(selected),'production_adoption':False,'analytic_branch_path':branches,
            'reference_input_to_optimizer':False,'fixed_root_torso_legs':True,
            'code_hashes':{p.name:sha(p) for p in [ROOT/'g1_pipeline/diagnose_arm_trajectory.py',ROOT/'g1_pipeline/gmr_arm_trajectory_seed.py',ROOT/'g1_pipeline/gmr_arm_branch_path.py']}}
    write_json(output/'design.json',design);results=[]
    for item in selected:
        row=rows[item['sample_id']];mid=row['source_motion_id']
        source=ROOT/'work/bones_arm_sequence_v10'/(mid+'.npy');q=np.load(source);before=q.copy()
        tp=ROOT/'work/bones_soma_rootweight_dev_v6'/(mid+'.time.json')
        times=np.array(json.loads(tp.read_text())['source']['target_timestamps'])
        frames,_=parse(ROOT/'work/bones_pairs_dev'/(mid+'.bvh'),{n:n for n in cfg['human_scale_table']},target_timestamps=times,rotation_basis='world_only')
        solver=create_retargeter(cfgpath,ROOT/'assets/project_g1_v1/g1.xml');m=solver.model;d=mujoco.MjData(m)
        parents_p=[];parents_r=[];targets=[]
        torso=m.body('torso_link').id
        for pose,frame in zip(q,frames):
            d.qpos[:]=pose;mujoco.mj_kinematics(m,d)
            parents_p.append(d.xpos[torso].copy());parents_r.append(d.xmat[torso].reshape(3,3).copy())
            solver.update_targets(frame,offset_to_ground=False)
            targets.append({k:(np.asarray(v[0]).copy(),np.asarray(v[1]).copy()) for k,v in solver.scaled_human_data.items()})
        parents_p=np.asarray(parents_p);parents_r=np.asarray(parents_r);reports=[]
        for side in ('left','right'):
            chain=ArmChain(m,side)
            if chain.parent!=torso:raise ValueError('UNEXPECTED_ARM_PARENT')
            bodies=[f'{side}_{name}_link' for name in ('shoulder_yaw','elbow','wrist_yaw')]
            names=[solver.ik_match_table2[b][0] for b in bodies]
            positions=np.array([[t[n][0] for n in names] for t in targets])
            quats=np.array([[t[n][1] for n in names] for t in targets])
            rotations=R.from_quat(quats.reshape(-1,4),scalar_first=True).as_matrix().reshape(len(q),3,3,3)
            weights=np.array([solver.human_body_to_task2[n].cost for n in names])
            initial=q[:,chain.adr]
            branch_report=None
            if branches:
                from .gmr_arm_branch_path import branch_path
                initial,branch_report=branch_path(chain,initial,parents_p,parents_r,positions,rotations,weights,step_weight=velocity_weight/.04)
            values,report=optimize(chain,initial,parents_p,parents_r,positions,rotations,weights,
                                   velocity_weight=velocity_weight,acceleration_weight=acceleration_weight)
            if branch_report:report['branch_path']=branch_report
            q[:,chain.adr]=values;reports.append({'side':side,**report})
            print(mid,side,{k:v for k,v in report.items() if k!='branch_path'},flush=True)
        path=output/(mid+'.npy');buf=io.BytesIO();np.save(buf,q);atomic_bytes(path,buf.getvalue())
        # Reference is evaluated only after the complete seed has been generated.
        ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
        ref,tt,_=resample(ref,rt,m.jnt_range[1:],end=np.nextafter(times[-1],np.inf))
        if q.shape!=ref.shape or not np.allclose(times,tt,atol=1e-10,rtol=0):raise ValueError('TIME_MISMATCH')
        result={'sample_id':row['sample_id'],'status':'MEASURED_WARM_START_NOT_FINAL_GMR',
                'frames':len(q),'solver_reports':reports,'seed_sha256':sha(path),'input_sha256':sha(source),
                'hinge_mae_deg':float(np.rad2deg(abs(q[:,7:]-ref[:,7:])).mean()),
                'before_hinge_mae_deg':item['hinge_mae_deg'],'after_continuity':metrics(m,q,times),
                'before_continuity':item['candidate_continuity'],'reference_continuity':item['reference_continuity'],
                'max_arm_correction_deg':float(np.rad2deg(abs(q[:,22:]-before[:,22:])).max())}
        results.append(result)
        write_json(ROOT/'audit'/f'{namespace}.json',{'design':design,'results':results})
        print(mid,'MAE',result['hinge_mae_deg'],'max_speed',max(result['after_continuity']['hinge_speed_max_rad_s']),flush=True)
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--branches',action='store_true');p.add_argument('--strong',action='store_true');a=p.parse_args();run(a.branches,a.strong)
