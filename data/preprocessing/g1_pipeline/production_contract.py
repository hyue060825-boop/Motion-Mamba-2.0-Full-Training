"""One G1 exporter/loader/representation check used by production candidates."""
import io
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,write_json,write_jsonl,sha,atomic_bytes,setup_imports
from .robot import compile_target,export_csv,read_csv,official_load,encode_csv,FORMAT,BASIS


def save_array(path,array):
    stream=io.BytesIO();np.save(stream,array);atomic_bytes(path,stream.getvalue())


def export_and_check(directory,sample_id,q,t,provenance,captions):
    setup_imports()
    import torch
    from motion_mamba2.skeleton.kinematics import fk
    from motion_mamba2.exports.mujoco import MujocoQposConverter
    from motion_mamba2.representation import MotionMamba2MotionRep
    from motion_data.humanml3d import HumanML3DMotionMamba2Dataset
    directory.mkdir(parents=True,exist_ok=True)
    csvpath=directory/'motion.csv';qpath=directory/'qpos.npy';fpath=directory/'features.npy'
    export_csv(csvpath,q,t,provenance);save_array(qpath,q)
    readback,rt,fps=read_csv(csvpath,format_id=FORMAT)
    p,r,s=official_load(csvpath);converter=MujocoQposConverter(s)
    direct=converter.to_qpos(torch.tensor(r).float()[None],torch.tensor(p).float()[None])[0].numpy()
    raw,rep,_=encode_csv(csvpath);inv=rep.inverse(raw,is_normalized=False);recovered=np.asarray(converter.dict_to_qpos(inv)[0])
    rotations,positions,_=fk(torch.tensor(r).float()[None],torch.tensor(p).float()[None],s)
    m,asset=compile_target();data=mujoco.MjData(m);si=[s.bone_order_names.index(m.joint(j).name.replace('_joint','_skel')) for j in range(1,m.njnt)]
    body=m.jnt_bodyid[1:];fkpos=0.;fkrot=0.
    for i,pose in enumerate(q):
        data.qpos[:]=pose;mujoco.mj_forward(m,data)
        expected_p=data.xpos[body]@BASIS.T
        expected_r=BASIS@data.xmat[body].reshape(-1,3,3)@BASIS.T
        actual_p=positions[0,i,si].numpy();actual_r=rotations[0,i,si].numpy()
        fkpos=max(fkpos,float(np.linalg.norm(expected_p-actual_p,axis=1).max()))
        fkrot=max(fkrot,float(np.rad2deg(R.from_matrix(expected_r@actual_r.swapaxes(-1,-2)).magnitude()).max()))
    canonical,_,transform=encode_csv(csvpath,canonicalize=True)
    restored=rep.translate_2d(canonical,torch.tensor(transform['planar_translation'])[None])
    restored=rep.rotate_to(restored,torch.tensor(transform['heading']))
    canonical_recovered=np.asarray(converter.dict_to_qpos(rep.inverse(restored,is_normalized=False))[0])
    save_array(fpath,canonical[0].numpy().astype(np.float32))
    contract=json.loads((ROOT/'contracts/motion_mamba2_414.json').read_text());stats=contract['chosen_normalizer']
    if sha(Path(stats)/'mean.npy')!=contract['mean_hash'] or sha(Path(stats)/'std.npy')!=contract['std_hash']:
        raise ValueError('FROZEN_G1_NORMALIZER_CHANGED')
    normalizer=MotionMamba2MotionRep(s,fps=25,stats_path=stats);norm=normalizer.normalize(canonical)
    denorm=normalizer.unnormalize(norm)
    manifest=directory/'project_loader.jsonl'
    write_jsonl(manifest,[{'id':sample_id,'feature_path':str(fpath),'frames':len(q),'captions':captions}])
    loader=HumanML3DMotionMamba2Dataset(manifest,stats,frames=len(q),horizon=52,control_probability=0,condition_dropout=0)
    batch=loader.sample(0,0);loader_error=float(abs(batch['motion']-norm[0]).max())
    def pose_error(value):
        return {'translation_max_m':float(abs(value[:,:3]-q[:,:3]).max()),
                'orientation_max_deg':float(np.rad2deg((R.from_quat(q[:,3:7],scalar_first=True).inv()*R.from_quat(value[:,3:7],scalar_first=True)).magnitude()).max()),
                'hinge_max_deg':float(np.rad2deg(abs(value[:,7:]-q[:,7:])).max())}
    metric={name:pose_error(value) for name,value in [('csv',readback),('official_loader',direct),('inverse',recovered),('canonical_inverse',canonical_recovered)]}
    gp=positions[0].numpy();speed=np.diff(gp,axis=0)*25;speed=np.concatenate([speed,speed[-1:]])
    contacts=(np.linalg.norm(speed[:,s.foot_joint_idx],axis=-1)<.15)&(gp[:,s.foot_joint_idx,1]<.10)
    skating={s.bone_order_names[i]:float(np.percentile(np.linalg.norm(speed[:,i][:,[0,2]],axis=1)[contacts[:,j]],95)) if contacts[:,j].any() else None for j,i in enumerate(s.foot_joint_idx)}
    expected_position=positions[:,:,1:].clone();expected_position[...,0]-=positions[:,:,:1,0];expected_position[...,2]-=positions[:,:,:1,2]
    position_error=float(abs(raw[:,:,5:104]-expected_position.reshape(1,len(q),99)).max())
    velocity_error=float(abs(raw[0,:,308:410]-torch.tensor(speed.reshape(len(q),102))).max())
    limits={'translation_max_m':1e-4,'orientation_max_deg':.02,'hinge_max_deg':.02}
    passed=all(v[k]<=limit for v in metric.values() for k,limit in limits.items())
    excess=float(max(0,np.max(m.jnt_range[1:,0]-q[:,7:]),np.max(q[:,7:]-m.jnt_range[1:,1])))
    finite=all(np.isfinite(a).all() for a in [q,canonical.numpy(),norm.numpy(),gp])
    details={'pose_roundtrips':metric,'fk_position_max_m':fkpos,'fk_orientation_max_deg':fkrot,
      'inverse_position_max_m':float(abs(inv['posed_joints'][0]-positions[0]).max()),
      'canonical_roundtrip_feature_max_abs':float(abs(restored-raw).max()),
      'position_block_max_m':position_error,'velocity_block_max_m_s':velocity_error,
      'contact_mismatches':int(np.sum(raw[0,:,410:414].numpy()!=contacts)),
      'project_loader_max_abs':loader_error,'project_loader_length':int(batch['length']),
      'normalization_allclose':bool(torch.allclose(canonical,denorm,atol=1e-5,rtol=1e-5)),
      'hinge_limit_excess_max_rad':excess,'finite':bool(finite),'fps':fps,'frames':len(q),
      'feature_shape':list(canonical[0].shape),'canonical_transform':transform,'skating_p95_m_s':skating,
      'contact_positive_frames':dict(zip([s.bone_order_names[i] for i in s.foot_joint_idx],contacts.sum(axis=0).tolist())),
      'asset_hash':asset['asset_hash'],'normalizer_hashes':{'mean':contract['mean_hash'],'std':contract['std_hash']}}
    passed=passed and finite and excess<=1e-4 and fkpos<=.001 and fkrot<=.1 and position_error<=.001 and velocity_error<=1e-5
    passed=passed and details['inverse_position_max_m']<=.001 and loader_error<=1e-5 and details['project_loader_length']==len(q)
    passed=passed and details['normalization_allclose'] and details['contact_mismatches']==0 and canonical.shape==(1,len(q),414)
    passed=passed and fps==25 and np.array_equal(rt,t)
    details['status']='PASS' if passed else 'FAIL';write_json(directory/'contract.json',details)
    artifacts={'csv_path':csvpath,'sidecar_path':csvpath.with_suffix('.json'),'qpos_path':qpath,'features_path':fpath,
               'contract_path':directory/'contract.json','project_loader_manifest':manifest}
    return details,{key:str(path.relative_to(ROOT)) for key,path in artifacts.items()}, {str(path.relative_to(ROOT)):sha(path) for path in artifacts.values()}
