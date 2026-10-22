"""Per-pair numeric, physical-time, contact and collision development metrics."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,XML,MESHES,read_jsonl,write_json,sha,setup_imports
from .robot import compile_target,read_csv,export_csv,official_load,encode_csv,FORMAT
from .resample import qpos as resample_qpos


def scene():
    # The locked XML already includes a floor in a second worldbody element,
    # with explicit foot/floor contact pairs. Do not append a duplicate plane.
    model,_=compile_target()
    if model.geom('floor').type[0]!=mujoco.mjtGeom.mjGEOM_PLANE:raise ValueError('EXPECTED_LOCKED_FLOOR')
    return model


def contacts_and_contract(path,q,t):
    setup_imports();import torch
    from motion_mamba2.skeleton.kinematics import fk
    from motion_mamba2.exports.mujoco import MujocoQposConverter
    export_csv(path,q,t,{'role':'paired_development_qa','raw_qpos_preserved':True})
    features,rep,_=encode_csv(path)
    p,r,s=official_load(path)
    _,gp,_=fk(torch.from_numpy(r).float()[None],torch.from_numpy(p).float()[None],s)
    gp=gp[0].numpy();speed=np.linalg.norm(np.diff(gp,axis=0),axis=-1)*25
    speed=np.concatenate([speed,speed[-1:]])
    contact=(speed[:,s.foot_joint_idx]<.15)&(gp[:,s.foot_joint_idx,1]<.10)
    inverse=rep.inverse(features,is_normalized=False)
    recovered=MujocoQposConverter(s).dict_to_qpos(inverse)[0]
    if hasattr(recovered,'numpy'):recovered=recovered.numpy()
    recovered=np.asarray(recovered)
    err={'frames':len(q),'feature_width':int(features.shape[-1]),'finite':bool(np.isfinite(recovered).all() and torch.isfinite(features).all()),
         'inverse_translation_max_m':float(np.max(np.abs(q[:,:3]-recovered[:,:3]))),
         'inverse_rotation_max_deg':float(np.rad2deg((R.from_quat(q[:,3:7],scalar_first=True).inv()*R.from_quat(recovered[:,3:7],scalar_first=True)).magnitude()).max()),
         'inverse_hinge_max_deg':float(np.rad2deg(np.abs(q[:,7:]-recovered[:,7:])).max()),
         'encoder_contact_mismatches':int(np.sum(features[0,:,410:414].numpy()!=contact))}
    err['status']='PASS' if err['finite'] and err['feature_width']==414 and err['inverse_translation_max_m']<=1e-4 and err['inverse_rotation_max_deg']<=.02 and err['inverse_hinge_max_deg']<=.02 and err['encoder_contact_mismatches']==0 else 'FAIL'
    foot_velocity=np.diff(gp[:,s.foot_joint_idx][:,:,[0,2]],axis=0)*25
    foot_velocity=np.concatenate([foot_velocity,foot_velocity[-1:]])
    speed_xy=np.linalg.norm(foot_velocity,axis=-1)
    skating={s.bone_order_names[i]:float(np.percentile(speed_xy[:,j][contact[:,j]],95)) if contact[:,j].any() else None for j,i in enumerate(s.foot_joint_idx)}
    return contact,err,skating,[s.bone_order_names[i] for i in s.foot_joint_idx]


def replay(m,q):
    d=mujoco.MjData(m);floor=m.geom('floor').id
    ground=[];self_depth=[];positions=[];rotations=[];pairs={}
    names=['left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link','torso_link']
    ids=[m.body(n).id for n in names]
    for x in q:
        d.qpos[:]=x;mujoco.mj_forward(m,d);g=s=0.
        for c in d.contact:
            depth=max(0.,-float(c.dist))
            if floor in (c.geom1,c.geom2):g=max(g,depth)
            else:
                s=max(s,depth)
                if depth>.01:
                    b1=m.body(int(m.geom_bodyid[c.geom1])).name;b2=m.body(int(m.geom_bodyid[c.geom2])).name
                    key=' | '.join(sorted((b1,b2)));pairs[key]=pairs.get(key,0)+1
        ground.append(g);self_depth.append(s);positions.append(d.xpos[ids].copy());rotations.append(d.xmat[ids].reshape(-1,3,3).copy())
    g=np.asarray(ground);s=np.asarray(self_depth)
    return np.asarray(positions),np.asarray(rotations),{
        'ground_penetration_p95_m':float(np.percentile(g,95)),'ground_penetration_max_m':float(g.max()),
        'raw_self_collision_depth_over_1cm_frame_fraction':float(np.mean(s>.01)),
        'raw_self_collision_max_depth_m':float(s.max()),'raw_self_contact_pair_counts':pairs,
        'self_collision_scope':'MuJoCo compiled collision masks/parent filtering; task-specific permitted contacts not yet frozen',
        'self_collision_release_metric':None,'self_collision_release_reason':'allowed-contact policy and supported scenes need development validation'}


def run(namespace='bones_soma_rest_smoke_v3',panel='smoke',sample_ids=None):
    if panel not in ('smoke','dev','accept'):raise ValueError('INVALID_PANEL')
    if panel=='accept':
        from .calibration_protocol import require_frozen
        require_frozen()
    root=ROOT/'work'/namespace;manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl'
    m=scene();base,_=compile_target();rows=[]
    run_report=json.loads((ROOT/'qa/aggregate'/f'{namespace}.json').read_text())
    executed={r['sample_id']:r for r in run_report['results']}
    panel_rows=read_jsonl(manifest)
    if sample_ids is not None:
        if not set(sample_ids)<={row['sample_id'] for row in panel_rows}:raise ValueError('SUBSET_OUTSIDE_FIXED_PANEL')
        selected=[row for row in panel_rows if row['sample_id'] in set(sample_ids)]
    else:selected=panel_rows
    for row in selected:
        result={'sample_id':row['sample_id'],'status':'ERROR'}
        try:
            mid=row['source_motion_id'];q=np.load(root/(mid+'.npy'));execution=executed[row['sample_id']]
            t=np.array(json.loads((root/(mid+'.time.json')).read_text())['source']['target_timestamps'])
            ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
            ref,_,_=resample_qpos(ref,rt,base.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
            ca,cc,cs,names=contacts_and_contract(root/'contracts'/(mid+'.candidate.csv'),q,t)
            ra,rc,rs,_=contacts_and_contract(root/'contracts'/(mid+'.reference.csv'),ref,t)
            cp,cr,cg=replay(m,q);rp,rr,rg=replay(m,ref)
            align=execution['fixed_planar_alignment'];rotation=R.from_euler('z',align['yaw_rad']);translation=np.asarray(align['translation'])
            aligned=rotation.apply(cp.reshape(-1,3)).reshape(cp.shape)+translation
            errors=np.linalg.norm(aligned-rp,axis=-1)
            orientation=R.from_matrix(rr[:,-1]).inv()*rotation*R.from_matrix(cr[:,-1])
            agreement={}
            for j,name in enumerate(names):
                a,b=ca[:,j],ra[:,j];tp=int(np.sum(a&b));tn=int(np.sum(~a&~b));fp=int(np.sum(a&~b));fn=int(np.sum(~a&b))
                f1=2*tp/(2*tp+fp+fn) if np.any(b) else None
                ba=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None
                agreement[name]={'f1':f1,'balanced_accuracy':ba,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
                                 'missing_reason':None if f1 is not None and ba is not None else 'reference lacks positive or negative support'}
            result.update(status='MEASURED',frames=len(q),candidate_contract=cc,reference_contract=rc,
                candidate_geometry=cg,reference_geometry=rg,candidate_skating_p95_m_s=cs,reference_skating_p95_m_s=rs,
                contact_agreement=agreement,extremity_world_aligned_mean_m=dict(zip(['left_foot','right_foot','left_wrist','right_wrist'],errors[:,:4].mean(axis=0).tolist())),
                extremity_root_relative_mean_m=dict(zip(['left_foot','right_foot','left_wrist','right_wrist'],np.linalg.norm(rotation.apply((cp-q[:,:3,None].transpose(0,2,1)).reshape(-1,3)).reshape(cp.shape)-(rp-ref[:,:3,None].transpose(0,2,1)),axis=-1)[:,:4].mean(axis=0).tolist())),
                torso_orientation_mean_deg=float(np.rad2deg(orientation.magnitude()).mean()),
                root_orientation_raw_mean_deg=float(np.rad2deg((R.from_quat(ref[:,3:7],scalar_first=True).inv()*R.from_quat(q[:,3:7],scalar_first=True)).magnitude()).mean()),
                per_joint_speed_p95_rad_s=np.percentile(np.abs(np.diff(q[:,7:],axis=0)*25),95,axis=0).tolist(),
                hardware_velocity_gate=None,hardware_velocity_reason='matched controller/motor envelope not yet verified',
                source_props=row.get('props'),body_position=row.get('body_position'),source_category=row.get('category'),
                candidate_sha256=sha(root/(mid+'.npy')))
        except Exception as exc:result['error']=repr(exc)
        rows.append(result)
        write_json(root/'pair_qa.json',{'scope':'development measurements, incomplete release metrics','manifest_sha256':sha(manifest),'expected':len(panel_rows),'selected':len(selected),'attempted':len(rows),'results':rows})
        print(result['sample_id'],result['status'],result.get('error',''),flush=True)
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--namespace',default='bones_soma_rest_smoke_v3');p.add_argument('--panel',default='smoke');a=p.parse_args()
    run(a.namespace,a.panel)
