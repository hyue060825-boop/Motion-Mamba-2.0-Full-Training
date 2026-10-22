from __future__ import annotations
from motion_mamba2.paths import artifact_path
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from .common import *
from .robot import compile_target, header, FORMAT, BASIS, official_load, read_csv, export_csv, encode_csv

STATS = Path(artifact_path('bones_seed_motion_mamba2_train_3c9f3e1238'))

def command(args):
    p = subprocess.run(args, text=True, capture_output=True, timeout=30)
    return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}

def environment():
    setup_imports()
    paths = {'run_root':str(ROOT),'motionmamba':str(REPO),'bones':str(BONES),'hiphi':str(HIPHI),
             'humanml3d':str(HML),'amass':str(HML/'amass_data'),'body_models':str(HML/'body_models'),
             'target_xml':str(XML),'target_meshes':str(MESHES),'legacy_stats':str(STATS),
             'gmr':str(ROOT/'deps/GMR'),'controller_repository':artifact_path('gr00t-wholebodycontrol_cb55417b96'),
             'hf_endpoint':'https://hf-mirror.com','github_fallback_proxy':'http://127.0.0.1:10408'}
    write_json(ROOT/'audit/paths.resolved.json',paths)
    versions = {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    write_json(ROOT/'audit/versions.lock.json',versions)
    git = {name:command(['git','-C',str(p),'rev-parse','HEAD']) for name,p in [('project',REPO),('humanml3d',HML),('controller',Path(paths['controller_repository']))]}
    patch = subprocess.run(['git','-C',str(REPO),'diff','--binary','HEAD'],capture_output=True,check=True).stdout
    atomic_bytes(ROOT/'audit/project_dirty.patch',patch)
    git['project_status'] = command(['git','-C',str(REPO),'status','--short'])
    git['dirty_patch_hash'] = sha(ROOT/'audit/project_dirty.patch')
    selected = list((REPO/'ardy/ardy').rglob('*.py')) + list((REPO/'mamba-transformer-keneral/src').rglob('*.py'))
    hashes = {str(p):sha(p) for p in selected}
    write_json(ROOT/'audit/project_source_hashes.json',hashes)
    write_json(ROOT/'audit/environment.json',{'time':now(),'python':os.sys.executable,'platform':platform.platform(),
        'git':git,'disk':{p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/data3','/data4']},
        'gpu':command(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv']),
        'memory':command(['free','-b']),'agent_instructions':'No AGENTS.md found in workspace ancestors or located project trees',
        'read_only_sources':True,'project_source_hash':digest(hashes)})
    controllers = []
    base = Path(paths['controller_repository'])
    for p in base.rglob('*.onnx'):
        controllers.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    write_json(ROOT/'audit/controller_assets.json',{'assets':controllers,'dynamic_qa_status':'NOT_RUN',
        'compatibility':'UNVERIFIED: target XML nu=0; requires separately audited actuated scene and exact kinematic mapping',
        'exposure':None,'exposure_reason':'pretrained controller source-motion exposure not established'})
    (ROOT/'audit/repository_audit.md').write_text(
        '# 实际仓库审计\n\n原仓库只读。工作区已有修改与删除，完整状态保存在 environment.json；本任务未修改它。\n\n'
        '旧 BonesSeedG1Converter 固定 120→25 nearest-frame；新版独立 exporter 的 sidecar 必须声明 25 FPS，'
        'encode_csv 只接受 25 FPS，不再采样。旧缓存未覆盖。\n\n'
        '官方 load_csv_g1_animation 的 retargeted_from_gmr 分支是 Frame+厘米+xyz 外禀角度+29 角度；'
        'retargeted_from_unitree 是无头、米、xyzw、rad。MujocoQposConverter.save_csv 默认 wxyz raw qpos 不能直接冒充任一带头格式。\n\n'
        'G1 XML 的网格缺失已用本机官方备份路径解析；未更改物理 XML，编译模型 nq=36,nv=35,nu=0。'
        '无 actuator 的模型可用于几何 QA，不能直接作为动力学跟踪场景。\n\n'
        '414 特征使用 [0:5]+[9:418] stats；local position 仍含 root height。'
        '训练入口类名 HumanML3DMotionMamba2Dataset 并不保证 human/G1 隔离，新 wrapper 强制 34/29/414/25 与 manifest hash。\n')
    return paths

def contract_definitions():
    setup_imports()
    import mujoco
    from motion_mamba2.skeleton import G1Skeleton34
    from motion_mamba2.representation import MotionMamba2MotionRep
    from scipy.spatial.transform import Rotation
    model, assets = compile_target()
    skel = G1Skeleton34(load=True)
    joints = []
    for j in range(model.njnt):
        joints.append({'id':j,'name':model.joint(j).name,'type':int(model.jnt_type[j]),
            'qpos_address':int(model.jnt_qposadr[j]),'dof_address':int(model.jnt_dofadr[j]),
            'axis':model.jnt_axis[j].tolist(),'range_rad':model.jnt_range[j].tolist(),
            'limited':bool(model.jnt_limited[j]),'position_in_body':model.jnt_pos[j].tolist(),
            'body_id':int(model.jnt_bodyid[j])})
    bodies = [{'id':i,'name':model.body(i).name,'parent':int(model.body_parentid[i]),
               'rest_pos':model.body_pos[i].tolist(),'rest_quat_wxyz':model.body_quat[i].tolist(),
               'mass':float(model.body_mass[i]),'inertia':model.body_inertia[i].tolist()} for i in range(model.nbody)]
    geoms = [{'id':i,'name':model.geom(i).name,'type':int(model.geom_type[i]),'body':int(model.geom_bodyid[i]),
              'contype':int(model.geom_contype[i]),'conaffinity':int(model.geom_conaffinity[i]),
              'size':model.geom_size[i].tolist(),'pos':model.geom_pos[i].tolist(),
              'quat_wxyz':model.geom_quat[i].tolist()} for i in range(model.ngeom)]
    robot = {**assets,'nq':model.nq,'nv':model.nv,'nu':model.nu,'joints':joints,'bodies':bodies,
             'geoms':geoms,'sites':[{'name':model.site(i).name,'body':int(model.site_bodyid[i]),'pos':model.site_pos[i].tolist()} for i in range(model.nsite)],
             'qpos0':model.qpos0.tolist(),'root_quaternion':'wxyz','units':'m/rad','world':'right handed Z-up, X-forward',
             'mujoco_to_motion_mamba2_basis':BASIS.tolist(),'gmr_asset_comparison':None,'gmr_asset_comparison_reason':'GMR source dependency not yet resolved'}
    write_json(ROOT/'contracts/g1_robot.json',robot)
    assert header()[7:] == [j['name']+'_dof' for j in joints[1:]]
    write_json(ROOT/'contracts/bones_g1_csv.json',{'format_id':FORMAT,'header':header(),'schema_hash':digest(header()),
        'golden_path':str(GOLDEN),'golden_hash':sha(GOLDEN),'delimiter':',','frame_start':0,'empty_lines':'reject',
        'root_translation_unit':'cm','root_euler':{'sequence':'xyz','convention':'SciPy lower-case extrinsic active','unit':'degrees'},
        'hinge_unit':'degrees','export_precision':'12 significant decimal digits','qpos_format':'explicit wxyz/m/rad, never shape inferred',
        'official_source_fps':120,'fps_evidence':{'path':str(BONES/'README.md'),'sha256':sha(BONES/'README.md'),'field':'move_duration_frames @120fps'},
        'loader':{'function':'motion_mamba2.data_processing.skeleton.load_csv_g1_animation','format':'retargeted_from_gmr','returns_translation_unit':'cm; caller multiplies 0.01'},
        'production_fps':25,'sidecar_required':True,'time_policy':'[start,end), t<=last real sample, no extrapolation'})
    neutral = skel.neutral_joints.numpy()
    nodes=[]
    for i,name in enumerate(skel.bone_order_names):
        parent=int(skel.joint_parents[i]); matching=[j for j in joints if j['name'].replace('_joint','_skel')==name]
        nodes.append({'name':name,'parent':parent,'neutral_position':neutral[i].tolist(),
            'offset_motion_mamba2_m':(neutral[i]-(neutral[parent] if parent>=0 else 0)).tolist(),
            'physical_joint':matching[0]['name'] if matching else ('floating_base_joint' if i==0 else None),
            'role':'free-root' if i==0 else ('hinge-body' if matching else 'fixed FK endpoint')})
    write_json(ROOT/'contracts/g1_skeleton34.json',{'nbjoints':34,'physical_hinges':29,'fixed_endpoints':4,'root_nodes':1,
        'nodes':nodes,'joints_asset_hash':sha(Path(skel.folder)/'joints.p'),'units':'m','world':'Y-up,Z-forward',
        'rest_rotations':'local hinge rotations are C * (body_rest_rotation * axis_angle) * C.T; extra endpoints identity',
        'basis':BASIS.tolist(),'foot_channels':skel.foot_joint_names})
    rep=MotionMamba2MotionRep(skel,fps=25)
    write_json(ROOT/'contracts/motion_mamba2_414.json',{'dimension':414,'fps':25,'blocks':{k:[v.start,v.stop] for k,v in rep.slice_dict.items()},
        'positions':'root m; non-root root-relative X/Z but world Y retains root height; no heading-local position rotation',
        'heading':'atan2((right_hip-left_hip).z, -(right_hip-left_hip).x), encoded cos/sin',
        'rotation_6d':'first two columns, concatenated column vectors (verify geometry implementation)',
        'velocities':'world m/s, forward difference*25; last valid velocity repeats previous; mask lengths excludes padding',
        'contacts':{'nodes':skel.foot_joint_names,'speed_m_s':.15,'world_height_m':.10,'rule':'both strict less-than'},
        'inverse':'global 6D -> local rotations -> FK; primary positions from rotations',
        'normalization':'(x-mean)/sqrt(std^2+1e-5), after canonicalization; inverse first unnormalizes',
        'canonicalization':'single starting hip heading and planar root translation, absolute root height retained',
        'dual_root_stats_dim':418,'dual_root_blocks':{'global_root':[0,5],'local_root':[5,9],'body':[9,418]},
        'explicit_stats_indices':list(range(5))+list(range(9,418)),
        'local_root_semantics':['angular velocity','local planar velocity x','local planar velocity z','global root y'],
        'chosen_normalizer':str(STATS),'mean_hash':sha(STATS/'mean.npy'),'std_hash':sha(STATS/'std.npy'),
        'freeze_policy':'existing G1 tokenizer/decoder normalizer remains pinned; new stats diagnostic only',
        'tokenizer_reconstruction_new_sources':None,'tokenizer_reconstruction_reason':'downstream gates not run'})
    return robot

def numerical_contracts():
    setup_imports()
    import torch, mujoco
    from scipy.spatial.transform import Rotation
    from motion_mamba2.exports.mujoco import MujocoQposConverter
    from motion_mamba2.representation import MotionMamba2MotionRep
    from .resample import qpos as resample
    torch.set_num_threads(2)
    m,_=compile_target()
    outdir=ROOT/'work/contract_checks';outdir.mkdir(parents=True,exist_ok=True)
    base=np.tile(m.qpos0,(5,1));base[:,2]=.80
    cases={'static':base.copy(),'root_translation':base.copy(),'turn_90':base.copy(),
           'pi_crossing':base.copy(),'quaternion_sign':base.copy(),'left_foot':base.copy(),'right_foot':base.copy()}
    cases['root_translation'][:,:3]+=np.arange(5)[:,None]*np.array([.01,.02,.005])
    cases['turn_90'][:,3:7]=Rotation.from_euler('z',np.linspace(0,90,5)[:,None],degrees=True).as_quat(scalar_first=True)
    cases['pi_crossing'][:,3:7]=Rotation.from_euler('z',np.array([178,179,180,181,182])[:,None],degrees=True).as_quat(scalar_first=True)
    cases['quaternion_sign'][1::2,3:7]*=-1
    cases['left_foot'][:,10]=np.linspace(0,.5,5)
    cases['right_foot'][:,16]=np.linspace(0,.5,5)
    for j in range(29):
        q=base.copy(); amplitude=min(.1, m.jnt_range[j+1,1]*.5)
        q[:,7+j]=np.linspace(0,amplitude,5); cases[f'joint_{j:02d}_{m.joint(j+1).name}']=q
    raw,ts,_=read_csv(GOLDEN,format_id=FORMAT,fps=120)
    q,t,mapping=resample(raw,ts,m.jnt_range[1:]);cases['real_bones']=q
    write_json(outdir/'real_resample_mapping.json',mapping)
    records=[]
    for name,q in cases.items():
        path=outdir/f'{name}.csv';t=np.arange(len(q))/25
        export_csv(path,q,t,{'audit_case':name})
        reread,rt,_=read_csv(path,format_id=FORMAT)
        pos,local,s=official_load(path)
        cv=MujocoQposConverter(s)
        direct=cv.to_qpos(torch.tensor(local).float()[None],torch.tensor(pos).float()[None])[0].numpy()
        features,rep,_=encode_csv(path)
        inverse=rep.inverse(features,is_normalized=False)
        recovered=cv.dict_to_qpos(inverse)[0]
        from motion_mamba2.skeleton.kinematics import fk
        g,p,_=fk(torch.tensor(local).float()[None],torch.tensor(pos).float()[None],s)
        inv_pos=inverse['posed_joints'][0].numpy()
        fk_pos=[];fk_rot=[];d=mujoco.MjData(m)
        for frame in range(len(q)):
            d.qpos[:]=q[frame];mujoco.mj_forward(m,d)
            for j in range(1,m.njnt):
                si=s.bone_order_names.index(m.joint(j).name.replace('_joint','_skel'));bi=m.jnt_bodyid[j]
                fk_pos.append(float(np.linalg.norm(BASIS@d.xpos[bi]-p[0,frame,si].numpy())))
                expected=BASIS@d.xmat[bi].reshape(3,3)@BASIS.T
                actual=g[0,frame,si].numpy()
                fk_rot.append(float(Rotation.from_matrix(expected@actual.T).magnitude()*180/np.pi))
        normalized_rep=MotionMamba2MotionRep(s,fps=25,stats_path=str(STATS))
        norm=normalized_rep.normalize(features);denorm=normalized_rep.unnormalize(norm)
        expected_vel=(p[:,1:]-p[:,:-1])*25
        expected_vel=torch.cat([expected_vel,expected_vel[:,-1:]],dim=1).reshape(1,len(q),102)
        expected_position=p[:,:,1:].clone();expected_position[...,0]-=p[:,:,:1,0];expected_position[...,2]-=p[:,:,:1,2]
        speeds=torch.linalg.vector_norm(p[:,1:]-p[:,:-1],dim=-1)*25;speeds=torch.cat([speeds,speeds[:,-1:]],dim=1)
        contacts=((speeds[:,:,s.foot_joint_idx]<.15)&(p[:,:,s.foot_joint_idx,1]<.10)).float()
        def rot_error(a,b):
            return float((Rotation.from_quat(a[:,3:7],scalar_first=True).inv()*Rotation.from_quat(b[:,3:7],scalar_first=True)).magnitude().max()*180/np.pi)
        metrics={'csv_translation_max_m':float(abs(q[:,:3]-reread[:,:3]).max()),'csv_rotation_max_deg':rot_error(q,reread),
            'csv_hinge_max_deg':float(np.rad2deg(abs(q[:,7:]-reread[:,7:]).max())),
            'official_qpos_translation_max_m':float(abs(q[:,:3]-direct[:,:3]).max()),'official_qpos_rotation_max_deg':rot_error(q,direct),
            'official_qpos_hinge_max_deg':float(np.rad2deg(abs(q[:,7:]-direct[:,7:]).max())),
            'inverse_qpos_translation_max_m':float(abs(q[:,:3]-recovered[:,:3]).max()),'inverse_qpos_rotation_max_deg':rot_error(q,recovered),
            'inverse_qpos_hinge_max_deg':float(np.rad2deg(abs(q[:,7:]-recovered[:,7:]).max())),
            'fk_position_max_m':max(fk_pos),'fk_orientation_max_deg':max(fk_rot),
            'inverse_position_max_m':float(abs(inv_pos-p[0].numpy()).max()),
            'position_block_max_m':float(abs(features[:,:,5:104]-expected_position.reshape(1,len(q),99)).max()),
            'velocity_block_max_m_s':float(abs(features[:,:,308:410]-expected_vel).max()),
            'contact_mismatches':int((features[:,:,410:414]!=contacts).sum()),
            'normalize_denormalize_max':float(abs(features-denorm).max()),
            'normalization_allclose':bool(torch.allclose(features,denorm,atol=1e-5,rtol=1e-5)),
            'frames':len(q),'features_shape':list(features.shape),'finite':bool(torch.isfinite(features).all())}
        metrics['passed']=bool(metrics['finite'] and metrics['normalization_allclose'] and metrics['contact_mismatches']==0 and
            max(metrics[k] for k in ['csv_translation_max_m','official_qpos_translation_max_m','inverse_qpos_translation_max_m'])<=1e-4 and
            max(metrics[k] for k in ['csv_rotation_max_deg','official_qpos_rotation_max_deg','inverse_qpos_rotation_max_deg','csv_hinge_max_deg','official_qpos_hinge_max_deg','inverse_qpos_hinge_max_deg'])<=.02 and
            metrics['fk_position_max_m']<=.001 and metrics['fk_orientation_max_deg']<=.1 and metrics['inverse_position_max_m']<=.001 and
            metrics['position_block_max_m']<=.001 and metrics['velocity_block_max_m_s']<=1e-5)
        records.append({'case':name,**metrics})
        if name=='real_bones':
            np.save(outdir/'real_bones_414.npy',features[0].numpy())
            np.save(outdir/'real_bones_qpos.npy',q)
            # Compare actual project loader and its normalizer, with controls disabled for a deterministic physical check.
            from motion_data.humanml3d import HumanML3DMotionMamba2Dataset
            mp=outdir/'project_loader.jsonl'
            write_jsonl(mp,[{'id':'audit_real_bones','feature_path':str(outdir/'real_bones_414.npy'),'frames':len(q),'captions':['audit']}])
            ds=HumanML3DMotionMamba2Dataset(mp,STATS,frames=len(q),horizon=52,control_probability=0,condition_dropout=0)
            batch=ds.sample(0,0)
            loader_err=float(abs(batch['motion']-norm[0]).max())
            records.append({'case':'actual_project_loader','max_error':loader_err,'frames':int(batch['length']),
                            'passed':loader_err<=1e-5 and int(batch['length'])==len(q),'representation':414})
            from motion_data.bones_seed import BonesSeedG1Converter
            old=BonesSeedG1Converter().convert(GOLDEN)
            new,_,_=encode_csv(path,canonicalize=True)
            write_json(ROOT/'reports/nearest_vs_rotation_aware.json',{'legacy_shape':list(old.shape),'new_shape':list(new[0].shape),
                 'rms_all_features_mixed_units':float(torch.sqrt(torch.mean((old-new[0])**2))),
                 'per_block_rms':{k:float(torch.sqrt(torch.mean((old[:,v]-new[0,:,v])**2))) for k,v in rep.slice_dict.items()},
                 'note':'descriptive only; feature blocks use different physical units; historical cache not overwritten'})
    report={'status':'PASS' if all(x['passed'] for x in records) else 'FAIL','count':len(records),'cases':records,'time':now(),
            'scope':'numeric preflight on synthetic motions and one complete official BONES clip; not retarget calibration or population QA'}
    write_json(ROOT/'qa/aggregate/numerical_contracts.json',report)
    return report
