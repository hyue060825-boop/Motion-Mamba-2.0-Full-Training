"""Hash-bound metrics for the real52f/9s HumanML3D-G1 cohort; no long-GT fabrication."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,hashlib,csv,time
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
import numpy as np
sys.path.insert(0,artifact_path('evaluator_native_v2_a933d7fb8b'))
from native_eval.metrics import candidate_pools,retrieval,psd_sqrt,embedding_health
from native_eval.v2_statistics import fid_to_many
from evaluate_suite import R,MODELS,TASKS,rd,wr,sha

def save(p,a):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in a for k in r)));w.writeheader();w.writerows(a)
def unit(x):
    x=np.asarray(x,np.float64);norm=np.linalg.norm(x,axis=-1,keepdims=True);assert norm.min()>1e-12;return x/norm
def fid(a,b):
    a=unit(a);b=unit(b);value=float(fid_to_many(a,[b])[0])
    ca=np.cov(a,rowvar=False);cb=np.cov(b,rowvar=False)
    independent=max(0.,float(((a.mean(0)-b.mean(0))**2).sum()+np.trace(ca)+np.trace(cb)-2*np.linalg.svd(psd_sqrt(ca)@psd_sqrt(cb),compute_uv=False).sum()))
    assert abs(value-independent)<1e-7
    return value

cohort=[json.loads(s) for s in (R/'COHORT.jsonl').open()];N=len(cohort)
records=[dict(sample_id=r['id'],source_family_id=r['source_id'],caption=r['caption'],frames=52) for r in cohort]
pools={str(seed):candidate_pools(records,seed) for seed in [1701,2903,20261020]}
wr(R/'QUERY_POOLS.json',dict(source_ids=[r['source_id'] for r in cohort],pools=pools))
summary=[];physical=[];timings=[];windows=[];bindings=[];reference={};health=[]
for method in MODELS:
    d=R/'models'/method;assert rd(d/'COMPLETE.json')['passed']
    for task in TASKS:
        W=1 if task=='short52' else 4
        banks=[];ends=[];wm=[];seams=[];inputs=[];prefixes=[];real_wm=[];real_seams=[]
        for shard in range((N+31)//32):
            p=d/'evaluation'/method/task/f'shard{shard:02d}'
            done=rd(p/'COMPLETE.json');sc=rd(p/'WINDOW_SCORE_COMPLETE.json')
            assert done['files']['geometry.npz']==sc['binding']['geometry_sha256']==sha(p/'geometry.npz')
            assert sc['features_sha256']==sha(p/'WINDOW_EMBEDDINGS.npz')
            bank=dict(np.load(p/'WINDOW_EMBEDDINGS.npz'));banks.append(bank)
            rr=[json.loads(s) for s in (p/'per_sample.jsonl').open()]
            with np.load(p/'geometry.npz') as z:
                prefixes.append(z['initial_history_joints'].copy())
                for i,r in enumerate(rr):
                    ce=float(np.linalg.norm(z['generated_root'][i,-1,[0,2]].astype(np.float64)-np.asarray(r['goal_root_xz_m']))*100)
                    assert abs(ce-r['constraint_error_cm'])<.001
            ends.extend(rr);inputs.extend(rd(p/'INPUT_AUDIT.json')['records'])
            ph=rd(p/'WINDOW_PHYSICAL_METRICS.json')
            wm.extend(r for r in ph['window_metrics'] if r['role']=='generated' and r['window']==W)
            seams.extend(r for r in ph['seams'] if r['role']=='generated' and r['entering_window']==W)
            real_wm.extend(r for r in ph['window_metrics'] if r['role']=='target' and r['window']==W)
            real_seams.extend(r for r in ph['seams'] if r['role']=='target' and r['entering_window']==W)
            bindings.append(dict(method=method,task=task,shard=shard,geometry=sha(p/'geometry.npz'),features=sha(p/'WINDOW_EMBEDDINGS.npz')))
        bank={k:(banks[0][k] if k=='starts' else np.concatenate([b[k] for b in banks],axis=1 if banks[0][k].ndim==3 else 0)) for k in banks[0]}
        assert bank['source_ids'].tolist()==[r['source_id'] for r in cohort]
        assert bank['distribution_generated'].shape==(W,N,256)
        if task not in reference:
            reference[task]=(bank,inputs,np.concatenate(prefixes))
        else:
            ref,ri,rprefix=reference[task]
            for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:
                assert np.allclose(bank[k],ref[k],atol=1e-4,rtol=1e-5),(method,task,k)
            keys=['source_id','effective_goal_root_xz_m','mask_nonzero_coordinates','noise_seed']
            assert [[x[k] for k in keys] for x in inputs]==[[x[k] for k in keys] for x in ri],('input_pairing',method,task)
            assert np.allclose(np.concatenate(prefixes),rprefix,atol=3e-5,rtol=1e-6),('physical_prefix',method,task)
        def r3(role,w):
            return float(np.mean([100*np.mean(retrieval(bank[f'text_seed{s}'][w],bank[f'retrieval_{role}_seed{s}'][w],p)[1]<=3) for s in [1701,2903] for p in pools.values()]))
        F=fid(bank['distribution_real'][-1],bank['distribution_generated'][-1]);Q=r3('generated',-1)
        tt=d/'timing'/method/task/'shard00';a=[r for r in rd(tt/'TIMING_DETAIL.json')['rows'] if not r['warmup']]
        assert len(a)==32 and all(r['batch']==1 for r in a)
        if method.startswith('native'):
            assert rd(R/'EXECUTION_REPAIR_LATENCY.json')['passed']
            assert rd(tt/'LOAD.json')['masked_kernel_repair_mode']=='masked'
            assert all(r['graph_delta']['builds']==r['graph_delta']['fallbacks']==0 for r in a)
            assert all(r['graph_delta']['replays']==2+22*W for r in a),('incomplete_actual_mask_graph_coverage',method,task)
        assert sha(tt/'geometry.npz')==rd(tt/'COMPLETE.json')['files']['geometry.npz']
        ms=1000*float(np.median([r['elapsed_seconds'] for r in a]))/W
        summary.append(dict(model=method,task=task,N=N,CE_cm=float(np.mean([r['constraint_error_cm'] for r in ends])),FID_L2=F,
            R3_percent=Q,latency_ms_per_window=ms,SR20_percent=float(100*np.mean([r['success_20cm'] for r in ends])),
            benchmark='HumanML3D-G1-retargeted-heldout',terminal_local_frames=52,windows=W))
        support=sum(r['predicted_support_foot_frames'] for r in wm);bad=sum(r['sliding_foot_frames'] for r in wm);slots=sum(r['total_foot_frames'] for r in wm)
        physical.append(dict(model=method,task=task,N=N,support_conditional_slip_percent=100*bad/support if support else None,
            predicted_support_percent=100*support/slots,seam_velocity_m_s=float(np.mean([r['root_velocity_discontinuity_m_s'] for r in seams])) if seams else None,
            joint_second_difference_m_per_frame2=float(np.mean([r['joint_second_difference_m_per_frame2'] for r in wm]))))
        timings.append(dict(model=method,task=task,N=32,batch=1,sequence_p50_s=float(np.median([r['elapsed_seconds'] for r in a])),amortized_ms_per_window=ms,
            timing_receipt_sha256=sha(tt/'COMPLETE.json'),geometry_sha256=sha(tt/'geometry.npz'),timing_detail_sha256=sha(tt/'TIMING_DETAIL.json')))
        for w in range(W):windows.append(dict(model=method,task=task,window=w+1,N=N,FID_L2=fid(bank['distribution_real'][w],bank['distribution_generated'][w]),R3_percent=r3('generated',w)))
        np.savez_compressed(R/f'{method}_{task}_bank.npz',**bank)
        if method==MODELS[0]:
            realR=r3('real',-1)
            summary.append(dict(model='retarget',task=task,N=N,CE_cm=None,FID_L2=0.,R3_percent=realR,latency_ms_per_window=None,SR20_percent=None,
                benchmark='HumanML3D-G1-retargeted-heldout',terminal_local_frames=52,windows=W))
            support=sum(r['predicted_support_foot_frames'] for r in real_wm);bad=sum(r['sliding_foot_frames'] for r in real_wm);slots=sum(r['total_foot_frames'] for r in real_wm)
            physical.append(dict(model='retarget',task=task,N=N,support_conditional_slip_percent=100*bad/support if support else None,
                predicted_support_percent=100*support/slots,seam_velocity_m_s=float(np.mean([r['root_velocity_discontinuity_m_s'] for r in real_seams])) if real_seams else None,
                joint_second_difference_m_per_frame2=float(np.mean([r['joint_second_difference_m_per_frame2'] for r in real_wm]))))
            health.append(dict(task=task,N=N,real_R3_percent=realR,chance_R3_percent=9.375,
                distribution=embedding_health(unit(bank['distribution_real'][-1])),
                interpretation='Frozen evaluator transfer diagnostic; not canonical HumanML3D evaluator or evidence of cross-domain calibration',
                real_retrieval_above_chance=bool(realR>9.375)))
save(R/'SUMMARY.csv',summary);save(R/'PHYSICAL.csv',physical);save(R/'LATENCY.csv',timings);save(R/'WINDOW_METRICS.csv',windows)
wr(R/'EVALUATOR_TRANSFER_AUDIT.json',dict(rows=health,evaluator_unchanged=True,no_retraining=True,
    sample_count_below256_dimension=True,finite_sample_FID_requires_caution=True,not_directly_comparable_to_HiPHI_FID=True))
wr(R/'FINAL_AUDIT.json',dict(passed=True,models=MODELS,N=N,tasks=TASKS,bindings=bindings,
    inputs_paired=True,geometry_CE_independently_recomputed=True,feature_L2_FID_independently_checked=True,
    no_root_ADE=True,long20_40='not measured; no natural official-test reference available',time=time.time()))
import subprocess
subprocess.run([sys.executable,str(R/'render_results.py')],check=True)
