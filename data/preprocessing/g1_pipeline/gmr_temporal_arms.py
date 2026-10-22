"""GMR sequence solver with human-only, bounded temporal arm initialization."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .locked_target import initialize_root_from_human
from .gmr_weighted_convergence import retarget,converge,cost
from .gmr_arm_seed import arm_seed
from .gmr_arm_trajectory_seed import ArmChain,optimize
from .gmr_arm_branch_path import branch_path


def solve(solver,frames,settings,convergence):
    if len(frames)<3:raise ValueError('TEMPORAL_ARMS_REQUIRE_THREE_FRAMES')
    initialize_root_from_human(solver,frames[0])
    poses=[];seed_events=[];targets=[]
    for i,frame in enumerate(frames):
        retarget(solver,frame,convergence)
        seeded=False
        for side in ('left','right'):
            tasks=[solver.human_body_to_task2[solver.ik_match_table2[f'{side}_{p}_link'][0]] for p in ('shoulder_yaw','elbow','wrist_yaw')]
            if i==0 or cost(solver.configuration,tasks)>settings['seed_trigger_cost']:
                _,info=arm_seed(solver,side,max_nfev=settings['seed_max_nfev'])
                seed_events.append({'frame':i,**info});seeded=True
        if seeded:converge(solver,solver.tasks2,**convergence)
        poses.append(solver.configuration.q.copy())
        targets.append({k:(np.asarray(v[0]).copy(),np.asarray(v[1]).copy()) for k,v in solver.scaled_human_data.items()})
    q=np.array(poses);initial=q.copy();m=solver.model;d=mujoco.MjData(m);torso=m.body('torso_link').id
    parent_p=[];parent_r=[]
    for pose in q:
        d.qpos[:]=pose;mujoco.mj_kinematics(m,d)
        parent_p.append(d.xpos[torso].copy());parent_r.append(d.xmat[torso].reshape(3,3).copy())
    parent_p=np.array(parent_p);parent_r=np.array(parent_r);arm_reports=[]
    for side in ('left','right'):
        chain=ArmChain(m,side)
        if chain.parent!=torso:raise ValueError('UNSUPPORTED_ARM_PARENT')
        names=[solver.ik_match_table2[f'{side}_{n}_link'][0] for n in ('shoulder_yaw','elbow','wrist_yaw')]
        target_p=np.array([[t[n][0] for n in names] for t in targets])
        quat=np.array([[t[n][1] for n in names] for t in targets])
        target_r=R.from_quat(quat.reshape(-1,4),scalar_first=True).as_matrix().reshape(len(q),3,3,3)
        weights=np.array([solver.human_body_to_task2[n].cost for n in names])
        start,path_report=branch_path(chain,q[:,chain.adr],parent_p,parent_r,target_p,target_r,weights,
                                      step_weight=settings['velocity_weight']/.04)
        values,fit_report=optimize(chain,start,parent_p,parent_r,target_p,target_r,weights,
                                  velocity_weight=settings['velocity_weight'],acceleration_weight=settings['acceleration_weight'],
                                  max_nfev=settings['trajectory_max_nfev'])
        q[:,chain.adr]=values;arm_reports.append({'side':side,'path':path_report,'fit':fit_report})
    warm=q.copy()
    import mink
    weights=np.zeros(m.nv)
    for side in ('left','right'):weights[m.jnt_dofadr[ArmChain(m,side).ids]]=settings['posture_cost']
    posture=mink.PostureTask(m,cost=weights);objectives=[];residuals={b:[] for b in solver.ik_match_table2}
    for i,(seed,frame) in enumerate(zip(warm,frames)):
        solver.update_targets(frame,offset_to_ground=False);solver.configuration.update(seed);posture.set_target(seed)
        converge(solver,solver.tasks2+[posture],**convergence)
        q[i]=solver.configuration.q;objectives.append(cost(solver.configuration,solver.tasks2))
        data=solver.configuration.data
        for body,entry in solver.ik_match_table2.items():
            pos,quat=solver.scaled_human_data[entry[0]];bid=m.body(body).id
            residuals[body].append([float(np.linalg.norm(data.xpos[bid]-pos)),
                float(np.rad2deg((R.from_matrix(data.xmat[bid].reshape(3,3)).inv()*R.from_quat(quat,scalar_first=True)).magnitude()))])
    if not np.isfinite(q).all():raise ValueError('NONFINITE_TEMPORAL_GMR_OUTPUT')
    report={'seed_events':seed_events,'arms':arm_reports,'human_task2_cost_mean':float(np.mean(objectives)),
            'human_task2_cost_max':float(max(objectives)),
            'temporal_seed_max_hinge_correction_deg':float(np.rad2deg(abs(warm[:,7:]-initial[:,7:])).max()),
            'gmr_continuation_max_hinge_correction_deg':float(np.rad2deg(abs(q[:,7:]-warm[:,7:])).max()),
            'body_residuals':{b:{'position_mean_m':float(np.mean(v,axis=0)[0]),'position_p95_m':float(np.percentile(v,95,axis=0)[0]),
                               'orientation_mean_deg':float(np.mean(v,axis=0)[1]),'orientation_p95_deg':float(np.percentile(v,95,axis=0)[1])} for b,v in residuals.items()},
            'reference_input':False}
    return q,report
