"""Decoded-motion local features and physical window diagnostics for one saved shard."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,hashlib,argparse,fcntl
import numpy as np
import torch
ROOT=Path(artifact_path('evaluator_6923422114'))
GEN=ROOT/'generator';REL=Path(artifact_path('bones-g1-native-v2_adcc182e95'))
sys.path.insert(0,artifact_path('evaluator_native_v2_a933d7fb8b'))
from native_eval.v2_score import verify_release,model_for
from native_eval.encoding import encode_arrays
from native_eval.metrics import local_starts
sys.path.insert(0,artifact_path('_runtime_8df4cd05cf'))
from motion_mamba2.skeleton import G1Skeleton34
from motion_mamba2.representation.reps.motion_representation import MotionMamba2MotionRep
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();folder=Path(a.job)
    lock=(folder/'score.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    complete=json.loads((folder/'COMPLETE.json').read_text());assert complete['status']=='COMPLETE'
    receipt=complete['receipt'];command=json.loads((folder/'COMMAND.json').read_text());panel=Path(command['argv'][command['argv'].index('--panel')+1])
    # Resolve the immutable generation panel, not the later timing alias.
    if sha(panel)!=command['panel_sha256']:
        candidates=[p for p in panel.parent.glob(command['task']+'_*.jsonl') if sha(p)==command['panel_sha256']]
        assert candidates, 'Generation panel hash has no preserved match'
        panel=candidates[0]
    assert sha(panel)==command['panel_sha256']
    rows=list(map(json.loads,panel.open()));N=len(rows)
    cfg=json.loads((ROOT/'WINDOW_EVALUATOR_QUALIFICATION.json').read_text());assert cfg['status']=='PASS'
    bind=dict(generation_receipt_sha256=sha(folder/'COMPLETE.json'),geometry_sha256=sha(folder/'geometry.npz'),
              qualification_sha256=sha(ROOT/'WINDOW_EVALUATOR_QUALIFICATION.json'),score_source_sha256=sha(Path(__file__)))
    if (folder/'WINDOW_SCORE_COMPLETE.json').exists():
        old=json.loads((folder/'WINDOW_SCORE_COMPLETE.json').read_text());assert old['binding']==bind
        assert old['features_sha256']==sha(folder/'WINDOW_EMBEDDINGS.npz');print('SCORE_REUSE',folder,flush=True);return
    assert bind['geometry_sha256']==complete['files']['geometry.npz']
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    verify_release(REL)
    skdata=json.loads((REL/'SKELETON.json').read_text());sk=G1Skeleton34(load=False)
    sk.register_buffer('neutral_joints',torch.tensor(skdata['data'],dtype=torch.float32),persistent=False)
    assert list(sk.joint_parents.numpy())==skdata['parents'];rep=MotionMamba2MotionRep(sk,25);feet=list(sk.foot_joint_idx)
    with np.load(folder/'geometry.npz') as z:
        raw={k:np.array(z[k],copy=True) for k in ['generated_physical','target_physical','generated_joints','target_joints','generated_root','target_root']}
        ids=[str(s) for s in z['sample_ids']]
    assert ids==[r['panel_id'] for r in rows]
    decoded={};geo={};errors={}
    for role in ['generated','target']:
        x=torch.from_numpy(raw[role+'_physical']);length=torch.full((len(x),),x.shape[1],dtype=torch.long)
        with torch.inference_mode():
            g=rep.inverse(x,is_normalized=False)
            y=rep(g['local_rot_mats'],g['root_positions'],to_normalize=False,to_canonicalize=False,lengths=length)
        assert torch.equal(y[:,:,:3],x[:,:,:3]),'Decoded adaptation changed root positions'
        delta=float(np.max(abs(g['posed_joints'].numpy()-raw[role+'_joints'])));assert delta<3e-5
        errors[role]=delta;decoded[role]=y.numpy()
        geo[role]={k:v.numpy() for k,v in g.items() if k in ['posed_joints','local_rot_mats','foot_contacts','root_positions']}
    T=decoded['generated'].shape[1];starts=local_starts(T);W=len(starts);records=[];ga=[];ta=[];tx=[]
    with np.load(ROOT/'T5_FEATURES.npz') as z:text={str(c):v.copy() for c,v in zip(z['captions'],z['features'])}
    for w,s in enumerate(starts):
        for i,r in enumerate(rows):
            records.append(dict(sample_id=f'{r["panel_id"]}::w{w+1:02d}',source_family_id=r['source_id'],caption=r['caption'],frames=52))
            ga.append(decoded['generated'][i,s:s+52]);ta.append(decoded['target'][i,s:s+52]);tx.append(text[r['caption']])
    model=model_for(REL,'distribution','cpu');stats=dict(np.load(REL/'stats.npz'))
    hg=encode_arrays(model,records,ga,stats)[0];hr=encode_arrays(model,records,ta,stats)[0];del model
    packed=dict(distribution_generated=hg.reshape(W,N,256),distribution_real=hr.reshape(W,N,256),
                starts=np.array(starts),sample_ids=np.array(ids),source_ids=np.array([r['source_id'] for r in rows]),
                actor_ids=np.array([r['actor_uid'] for r in rows]),captions=np.array([r['caption'] for r in rows]))
    stats=dict(np.load(ROOT/'retrieval_stats.npz'))
    for seed in [1701,2903]:
        assert sha(ROOT/f'seed{seed}/retrieval.pt')==cfg['binding']['retrieval_checkpoints'][str(seed)]
        model=model_for(ROOT/f'seed{seed}','retrieval','cpu')
        _,mg,tt=encode_arrays(model,records,ga,stats,tx);_,mr,_=encode_arrays(model,records,ta,stats);del model
        packed[f'retrieval_generated_seed{seed}']=mg.reshape(W,N,256)
        packed[f'retrieval_real_seed{seed}']=mr.reshape(W,N,256)
        packed[f'text_seed{seed}']=tt.reshape(W,N,256)
    metrics=[];seams=[]
    for role in ['generated','target']:
        g=geo[role];j=g['posed_joints'];root=g['root_positions'];contacts=g['foot_contacts'];rot=g['local_rot_mats']
        for w,s in enumerate(starts):
            f=j[:,s:s+52,feet,:];speed=np.linalg.norm(np.diff(f[:,:,:,[0,2]],axis=1),axis=-1)*25
            c=contacts[:,s+1:s+52].astype(bool);bad=(speed>.15)&c
            jit=np.linalg.norm(np.diff(j[:,s:s+52],n=2,axis=1),axis=-1).mean(axis=(1,2))
            for i,r in enumerate(rows):
                support=int(c[i].sum());slots=c[i].size
                metrics.append(dict(role=role,panel_id=r['panel_id'],source_id=r['source_id'],actor_uid=r['actor_uid'],
                    window=w+1,start=s,end_exclusive=s+52,foot_sliding_proxy_percent=float(100*bad[i].sum()/slots),
                    predicted_support_fraction_percent=float(100*support/slots),
                    support_conditional_sliding_percent=float(100*bad[i].sum()/support) if support else None,
                    sliding_foot_frames=int(bad[i].sum()),predicted_support_foot_frames=support,total_foot_frames=slots,
                    joint_second_difference_m_per_frame2=float(jit[i])))
        for boundary in range(52,T,52):
            before=(root[:,boundary-1]-root[:,boundary-2])*25;after=(root[:,boundary]-root[:,boundary-1])*25
            step=np.linalg.norm(root[:,boundary]-root[:,boundary-1],axis=-1);jump=np.linalg.norm(after-before,axis=-1)
            trace=(rot[:,boundary-1]*rot[:,boundary]).sum(axis=(-1,-2));angle=np.degrees(np.arccos(np.clip((trace-1)/2,-1,1)))
            for i,r in enumerate(rows):
                seams.append(dict(role=role,panel_id=r['panel_id'],source_id=r['source_id'],actor_uid=r['actor_uid'],
                    entering_window=boundary//52+1,boundary_generated_frame=boundary,
                    root_interframe_step_m=float(step[i]),root_velocity_discontinuity_m_s=float(jump[i]),
                    local_joint_rotation_step_rms_deg=float(np.sqrt(np.mean(angle[i]**2))),local_joint_rotation_step_max_deg=float(angle[i].max())))
    old=[json.loads(s) for s in (folder/'per_sample.jsonl').read_text().splitlines()]
    if T>52:
        for i,r in enumerate(rows):
            values=[s['root_velocity_discontinuity_m_s'] for s in seams if s['role']=='generated' and s['panel_id']==r['panel_id']]
            assert np.allclose(values,old[i]['boundary_velocity_discontinuity_m_s'],rtol=1e-4,atol=1e-4)
    constraints=[]
    for i,r in enumerate(rows):
        assert r['goal_joints']==[0] and len(r['goal_frames'])==1
        frame=int(r['goal_frames'][0]);goal=np.array(r['fixed_goal_root_xz_m'])
        err=float(np.linalg.norm(geo['generated']['root_positions'][i,frame,[0,2]]-goal)*100)
        constraints.append(dict(panel_id=r['panel_id'],actor_uid=r['actor_uid'],source_id=r['source_id'],
            generation_window=frame//52+1,goal_frame=frame,goal_joint=0,constraint_error_cm=err,
            target_role='imposed fixed root goal; natural reference trajectory is not a required route'))
        assert abs(err-old[i]['constraint_error_cm'])<.001
    np.savez_compressed(folder/'WINDOW_EMBEDDINGS.npz',**packed)
    write(folder/'WINDOW_PHYSICAL_METRICS.json',dict(window_metrics=metrics,seams=seams,constraints=constraints,
        slip_definition='same historical proxy: horizontal foot speed>0.15m/s AND raw predicted contact>0.5, denominator all foot-frame slots',
        support_conditional_definition='same sliding numerator divided by predicted contact count; None when no contact',
        contacts_are_ground_truth=False,jitter_units='meters/frame^2, mean norm of joint second difference',
        seams_scope='actual generated52-frame boundaries; velocity difference is not a position jump or Viewer acceptance test'))
    done=dict(status='COMPLETE_WINDOW_FEATURES_NOT_AGGREGATE_SCORES',binding=bind,model=receipt['model'],task=receipt['task'],N=N,
        frames=T,windows=W,features_sha256=sha(folder/'WINDOW_EMBEDDINGS.npz'),metrics_sha256=sha(folder/'WINDOW_PHYSICAL_METRICS.json'),
        adapter=dict(root_unchanged_exact=True,max_saved_pose_delta_m=errors,foot_indices=feet,method='standard_rotation_decode_fk_reencode_v1; no GT-dependent correction'),time=time.time())
    write(folder/'WINDOW_SCORE_COMPLETE.json',done);verify_release(REL);print('WINDOW_FEATURES_COMPLETE',folder,N,W,flush=True)
if __name__=='__main__':main()
