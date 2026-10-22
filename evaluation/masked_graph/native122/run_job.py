"""Read-only goal compiler around the frozen reference sampler.

The reference sampler retains natural GT for diagnostics; its route-relative
scores are private, explicitly non-primary. Primary scoring below uses only
the imposed root target and its specified deadline. No model/rollout changes.
"""
import argparse,contextlib,fcntl,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
from util import ROOT,MODELS,TASKS,read,write,rows,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--variant',choices=['controlled','reference','identity','tainted_future'],default='controlled')
    a=parser.parse_args();plan=read(ROOT/'PLAN.json');info=read(ROOT/'CHECKPOINTS.json')[a.model];specification=plan['tasks'][a.task]
    assert read(ROOT/'PREPARED.json')['status']=='PASS_CPU_COHORT_GOALS_SEEDS_AND_CHECKPOINT_IDENTITIES'
    tag='pilot' if a.pilot else f'shard{a.shard:02d}'
    panel=ROOT/'panels'/f'{a.task}_{tag}.jsonl';entries=rows(panel)
    if not a.pilot:
        assert a.variant=='controlled' and read(ROOT/'GATES.json')['status']=='PASS_ALL_MODELS_REFERENCE_IDENTITY_AND_NO_FUTURE_GT_LEAK'
        assert sha(panel)==read(ROOT/'PHASE_PANEL.json')['sha256']
    if a.variant in ['reference','identity']:assert a.task=='aligned_w04'
    dest=ROOT/os.environ.get('RUN_PHASE','evaluation')/a.model/a.task/(a.variant if a.pilot else tag)
    dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'COMPLETE.json').exists():print('REUSE',dest,flush=True);return
    ck=Path(info['checkpoint']);st=ck.stat()
    assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns)==info['frozen_file_stat'],'Checkpoint file changed after full SHA256 verification'
    for name,digest in plan['prepared_hashes'].items():assert sha(ROOT/name)==digest,('Frozen input/sampler changed',name)
    os.environ['EVAL_TRAINING_ROOT']=info['training_root'];os.environ['EVAL_ARCH']='native' if a.model.startswith('native') else a.model;os.environ['EVAL_LABEL']=a.model
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    if a.model=='ardy':os.environ['MOTION_MAMBA2_SEQUENCE_MODEL']='transformer'
    sys.path.insert(0,info['training_root']);import common
    sys.path.insert(0,str(ROOT/'sampler'))
    import numpy as np,torch
    script=ROOT/'sampler/evaluate_motion_mamba2_official_table5.py'
    ms=importlib.util.spec_from_file_location('controlled_reference_entry',script);entry=importlib.util.module_from_spec(ms);ms.loader.exec_module(entry)
    from harness import prepare_entry
    prepare_entry(entry)
    original=entry._build_official_sample;audits=[];lookup={r['panel_id']:r for r in entries}
    def digest_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def build(dataset,index,augmentation_seed,**kw):
        actual=kw['panel_entry'];assert actual['goal_joints']==[0]
        f=int(actual['goal_frames'][0]);h=kw['history_frames'];total=kw['total_frames']
        base=dict(actual,goal_frames=[total-h-1],goal_joints=[0])
        out=original(dataset,index,augmentation_seed,**dict(kw,panel_entry=base))
        prefix=out['motion'][:h].clone();root=kw['motion_rep'].slice_dict['root_pos'];assert root.start==0
        if a.variant=='reference':
            effective=(out['motion'][h+f,[0,2]]*torch.sqrt(dataset.std[[0,2]].square()+1e-5)+dataset.mean[[0,2]]).tolist()
        else:
            mask=torch.zeros_like(out['motion_mask']);observed=torch.zeros_like(out['observed_motion'])
            mask[h+f,[root.start,root.start+2]]=1.
            if a.variant=='identity':
                observed[h+f,[0,2]]=out['motion'][h+f,[0,2]]
                effective=(out['motion'][h+f,[0,2]]*torch.sqrt(dataset.std[[0,2]].square()+1e-5)+dataset.mean[[0,2]]).tolist()
                assert torch.equal(mask,out['motion_mask']) and torch.equal(observed,out['observed_motion']),'Identity compiler differs from original'
            else:
                goal=torch.tensor(actual['fixed_goal_root_xz_m'],dtype=out['motion'].dtype)
                observed[h+f,[0,2]]=(goal-dataset.mean[[0,2]])/torch.sqrt(dataset.std[[0,2]].square()+1e-5)
                recovered=observed[h+f,[0,2]]*torch.sqrt(dataset.std[[0,2]].square()+1e-5)+dataset.mean[[0,2]]
                assert torch.allclose(recovered,goal,atol=2e-6,rtol=1e-6),'Goal normalizer roundtrip failed'
                effective=actual['fixed_goal_root_xz_m']
            out.update(observed_motion=observed,motion_mask=mask,goal_frames=torch.tensor([f],dtype=torch.long),goal_joints=torch.tensor([0],dtype=torch.long))
        assert out['motion_mask'].count_nonzero().item()==2
        assert out['observed_motion'][out['motion_mask']==0].count_nonzero().item()==0
        assert torch.equal(out['motion'][:h],prefix)
        if a.variant=='tainted_future':out['motion'][h:]=out['motion'][h:].flip(0)
        audits.append(dict(panel_id=actual['panel_id'],source_id=actual['id'],goal_frame=f,absolute_goal_frame=h+f,
            effective_goal_root_xz_m=effective,goal_normalized=out['observed_motion'][h+f,[0,2]].tolist(),
            mask_nonzero_coordinates=out['motion_mask'].nonzero().tolist(),mask_sha256=digest_tensor(out['motion_mask']),
            observed_sha256=digest_tensor(out['observed_motion']),prefix_sha256=digest_tensor(prefix),noise_seed=out['noise_seed'],
            actual_target_exposed_before_entering_current_window=True,only_root_XZ_observed=True,
            hidden_future_gt_sha256=digest_tensor(out['motion'][h:])))
        return out
    entry._build_official_sample=build
    mode={'ardy':'original','minimal':'minimal','native122':'hybrid','native110':'hybrid','native117':'hybrid'}[a.model]
    argv=['--mode',mode,'--model','g1','--dataset-name','HiPHI-controlled-goal-diagnostic-v1','--checkpoint',info['checkpoint'],
        '--manifest',str(ROOT/'manifest.jsonl'),'--stats-dir',info['stats_dir'],'--text-cache',str(ROOT/'text_cache'),
        '--samples',str(len(entries)),'--batch-size',os.environ.get('EVAL_BATCH_SIZE','2'),'--diffusion-steps','10','--cfg-weight-text','2','--cfg-weight-constraint','2',
        '--seed','20260831','--device','cuda:0','--panel',str(panel),'--attention-window-chunks','4',
        '--protocol','out_of_horizon','--total-seconds',str(specification['total_frames']/25),'--selection-min-seconds',str(1012/25),
        '--output',str(dest/'REFERENCE_GT_DIAGNOSTICS_NOT_PRIMARY.json'),'--artifacts-output',str(dest/'geometry.npz'),
        '--per-sample-output',str(dest/'REFERENCE_GT_DIAGNOSTICS_NOT_PRIMARY.jsonl')]
    receipt=dict(pid=os.getpid(),model=a.model,task=a.task,variant=a.variant,pilot=a.pilot,shard=a.shard,N=len(entries),cuda=os.getenv('CUDA_VISIBLE_DEVICES'),
        argv=[str(script)]+argv,checkpoint_sha256=info['sha256'],checkpoint_step=info['step'],panel_sha256=sha(panel),sampler_sha256=sha(script),compiler_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'PLAN.json'),time=time.time())
    write(dest/'COMMAND.json',receipt);started=time.monotonic()
    try:
        sys.argv=[str(script)]+argv
        with (dest/'reference_stdout.log').open('w') as log,contextlib.redirect_stdout(log):code=entry.main()
        assert code in [0,None],code
        assert len(audits)==len(entries)
        load=read(dest/'LOAD.json');assert load['global_step']==info['step'] and load['all_loaded_denoiser_weights_verified']
        write(dest/'INPUT_AUDIT.json',dict(status='PASS_EXACT_ROOT_ONLY_GOAL_COMPILATION',records=audits,training_or_sampling_changes=False))
        metrics=[]
        old=rows(dest/'REFERENCE_GT_DIAGNOSTICS_NOT_PRIMARY.jsonl')
        with np.load(dest/'geometry.npz') as z:
            assert [str(x) for x in z['sample_ids']]==[r['panel_id'] for r in entries]
            for k in ['generated_physical','target_physical','generated_joints','target_joints','generated_root','target_root']:assert np.isfinite(z[k]).all()
            assert np.allclose(z['generated_physical'][:,:,:3],z['generated_root'],atol=3e-5,rtol=1e-6)
            for i,(r,au,previous) in enumerate(zip(entries,audits,old)):
                f=r['goal_frames'][0];goal=np.array(au['effective_goal_root_xz_m']);path=z['generated_root'][i,:, [0,2]].T
                if path.shape!=(specification['generated_frames'],2):path=z['generated_root'][i][:,[0,2]]
                distances=np.linalg.norm(path-goal,axis=-1);ce=float(distances[f]*100);window_start=(f//52)*52
                before=distances[max(window_start-1,0)]
                metrics.append(dict(panel_id=r['panel_id'],source_id=r['source_id'],actor_uid=r['actor_uid'],caption=r['caption'],model=a.model,task=a.task,
                    bucket=r['difficulty_bucket'],goal_root_xz_m=goal.tolist(),goal_distance_m=float(np.linalg.norm(goal)),noise_seed=r['noise_seed'],goal_generated_frame=f,
                    goal_window=f//52+1,within_window_goal_frame=f%52,effective_current_window_sec=(f%52+1)/25,
                    goal_time_including_prefix_sec=(f+25)/25,generated_frames=specification['generated_frames'],
                    constraint_error_cm=ce,success_10cm=ce<=10,success_20cm=ce<=20,success_50cm=ce<=50,
                    distance_before_current_window_cm=float(before*100),distance_reduced_in_current_window_cm=float((before-distances[f])*100),
                    required_speed_at_current_window_entry_m_s=float(before/((f%52+1)/25)),
                    distance_to_fixed_goal_at_generated_window_ends_cm=[float(distances[min(s+51,f)]*100) for s in range(0,f+1,52)],
                    endpoint_only_score=True,no_intermediate_path_penalty=True,
                    foot_skate_percent_proxy=previous['foot_skate_percent_proxy'],boundary_velocity_discontinuity_m_s=previous['boundary_velocity_discontinuity_m_s'],
                    jitter_m_per_frame2=previous['jitter_m_per_frame2'],max_root_speed_m_s=previous['max_root_speed_m_s']))
        (dest/'per_sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in metrics))
        result=dict(status='COMPLETE_CONTROLLED_ENDPOINT_SCORING',samples=len(metrics),model=a.model,task=a.task,variant=a.variant,
            metric_scope='Only imposed physical root XZ target at specified frame; no Root ADE or intermediate route penalty',
            constraint_error_cm_mean=float(np.mean([x['constraint_error_cm'] for x in metrics])),constraint_error_cm_median=float(np.median([x['constraint_error_cm'] for x in metrics])),
            **{f'success_{c}cm':float(np.mean([x[f'success_{c}cm'] for x in metrics])) for c in [10,20,50]},
            frame_specification=specification,checkpoint=info['checkpoint'],checkpoint_step=info['step'],sampler_unchanged=True)
        write(dest/'result.json',result)
        write(dest/'COMPLETE.json',dict(status='COMPLETE',receipt=receipt,elapsed=time.monotonic()-started,files={n:sha(dest/n) for n in ['geometry.npz','per_sample.jsonl','result.json','LOAD.json','INPUT_AUDIT.json']},time=time.time()))
        print('JOB_COMPLETE',a.model,a.task,tag,a.variant,len(metrics),round(time.monotonic()-started,2),flush=True)
    except Exception:
        write(dest/'FAILURE.json',dict(receipt=receipt,error=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
