"""Complete paired cohort only: independent endpoint checks and local52 metrics."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import collections,concurrent.futures,csv,hashlib,json,os,sys,time
import numpy as np
from util import ROOT,MODELS,TASKS,read,write,rows,sha
sys.path.insert(0,artifact_path('evaluator_native_v2_a933d7fb8b'))
from native_eval.metrics import candidate_pools,retrieval
from native_eval.v2_statistics import fid_to_many
OUT=ROOT/'aggregate';OUT.mkdir(exist_ok=True)

def csvwrite(p,rr):
    if not rr:return
    keys=list(dict.fromkeys(k for r in rr for k in r));tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
    os.replace(tmp,p)

def cluster_draw(actor_ids,rng):
    actors=np.unique(actor_ids)
    return np.concatenate([((ix:=np.flatnonzero(actor_ids==a))[rng.integers(len(ix),size=len(ix))]) for a in rng.choice(actors,len(actors),replace=True)])

def bank(model,task):
    path=OUT/task;path.mkdir(exist_ok=True);dest=path/f'{model}_bank.npz';receipt=dest.with_suffix('.json')
    if receipt.exists():
        r=read(receipt);assert sha(dest)==r['sha256']
        for s in r['sources']:assert sha(Path(s['path']))==s['sha256']
        return dict(np.load(dest))
    plan=read(ROOT/'PLAN.json');parts=[];sources=[]
    for c in [x for x in plan['catalog'] if x['task']==task]:
        p=ROOT/'evaluation'/model/task/f'shard{c["shard"]:02d}';sr=read(p/'WINDOW_SCORE_COMPLETE.json')
        assert sr['N']==c['N'] and sha(p/'WINDOW_EMBEDDINGS.npz')==sr['features_sha256']
        sources.append(dict(path=str(p/'WINDOW_SCORE_COMPLETE.json'),sha256=sha(p/'WINDOW_SCORE_COMPLETE.json')))
        parts.append(dict(np.load(p/'WINDOW_EMBEDDINGS.npz')))
    starts=parts[0]['starts'];assert all(np.array_equal(z['starts'],starts) for z in parts)
    b={k:(starts if k=='starts' else np.concatenate([z[k] for z in parts],axis=1 if parts[0][k].ndim==3 else 0)) for k in parts[0]}
    assert len(b['sample_ids'])==506 and len(set(b['source_ids']))==506
    np.savez_compressed(dest,**b);write(receipt,dict(N=506,sha256=sha(dest),sources=sources));return b

def endpoints():
    allrows=[];audit=[];summary=[];contrasts=[]
    plan=read(ROOT/'PLAN.json');by_source={r['id']:r for r in rows(ROOT/'source_protocol/LOCOMOTION_CANDIDATES_GT_ONLY.jsonl')}
    actual_goals={}
    for sid,r in by_source.items():
        x=np.load(r['feature_path'],mmap_mode='r');actual_goals[sid]=np.array(x[r['crop_start']+499,[0,2]],np.float32)-np.array(x[r['crop_start']+23,[0,2]],np.float32)
    for task in TASKS:
        values={};order=None
        for model in MODELS:
            rr=[];max_delta=0.
            for c in [x for x in plan['catalog'] if x['task']==task]:
                d=ROOT/'evaluation'/model/task/f'shard{c["shard"]:02d}';complete=read(d/'COMPLETE.json');assert complete['status']=='COMPLETE'
                assert sha(d/'per_sample.jsonl')==complete['files']['per_sample.jsonl']
                rrs=rows(d/'per_sample.jsonl');pp=rows(Path(c['panel']));assert len(rrs)==c['N']
                with np.load(d/'geometry.npz') as z:
                    root=z['generated_root']
                    for i,(r,p) in enumerate(zip(rrs,pp)):
                        goal=actual_goals[r['source_id']];assert np.allclose(goal,p['fixed_goal_root_xz_m'],atol=1e-7,rtol=1e-7)
                        ce=float(np.linalg.norm(root[i,p['goal_frames'][0],[0,2]]-goal)*100)
                        delta=abs(ce-r['constraint_error_cm']);assert delta<.001;max_delta=max(max_delta,delta)
                        rr.append(r)
            assert len(rr)==506
            ids=[r['source_id'] for r in rr]
            if order is None:order=ids
            assert ids==order
            values[model]=rr;allrows+=rr
            audit.append(dict(task=task,model=model,N=506,max_recomputed_endpoint_delta_cm=max_delta,status='PASS_REAL_GOAL_REEXTRACTION_AND_ENDPOINT_GEOMETRY'))
        groups=['ALL','near_0.5_1.5m','medium_1.5_3m','far_3_6m']
        for group in groups:
            ix=np.array([i for i,r in enumerate(values[MODELS[0]]) if group=='ALL' or r['bucket']==group]);base=[values[MODELS[0]][i] for i in ix]
            actors=np.array([r['actor_uid'] for r in base]);rng=np.random.default_rng(int(hashlib.sha256(f'{task}:{group}'.encode()).hexdigest()[:12],16))
            draws=[cluster_draw(actors,rng) for _ in range(2000)]
            ce=np.stack([[values[m][i]['constraint_error_cm'] for i in ix] for m in MODELS])
            boot=np.array([[ce[:,di].mean(axis=1)] for di in draws]).reshape(2000,len(MODELS))
            for j,m in enumerate(MODELS):
                v=ce[j];lo,hi=np.quantile(boot[:,j],[.025,.975]);s=plan['tasks'][task]
                summary.append(dict(task=task,model=m,stratum=group,N_sources=len(ix),actors=len(set(actors)),goal_time_including_prefix_sec=s['goal_time_including_prefix_sec'],within_window_goal_frame=s['within_window_goal_frame'],current_window_remaining_sec=s['effective_current_window_frames']/25,
                    constraint_error_cm=float(v.mean()),CE_median_cm=float(np.median(v)),CE_p90_cm=float(np.quantile(v,.9)),CE_ci95_low=float(lo),CE_ci95_high=float(hi),
                    success_10cm=float(np.mean(v<=10)),success_20cm=float(np.mean(v<=20)),success_50cm=float(np.mean(v<=50)),
                    before_current_window_distance_cm=float(np.mean([values[m][i]['distance_before_current_window_cm'] for i in ix])),
                    distance_reduced_current_window_cm=float(np.mean([values[m][i]['distance_reduced_in_current_window_cm'] for i in ix])),
                    scope='paired fixed goal; single paired generator realization; actor then source bootstrap2000; no Root ADE'))
            for j,m in enumerate(MODELS):
                for k in range(j):
                    dd=boot[:,j]-boot[:,k];lo,hi=np.quantile(dd,[.025,.975])
                    contrasts.append(dict(task=task,stratum=group,model=m,reference=MODELS[k],N_sources=len(ix),mean_CE_difference_cm=float((ce[j]-ce[k]).mean()),paired_ci95_low=float(lo),paired_ci95_high=float(hi)))
    csvwrite(OUT/'ENDPOINT_PER_SOURCE.csv',[dict(r,goal_root_xz_m=json.dumps(r['goal_root_xz_m']),distance_to_fixed_goal_at_generated_window_ends_cm=json.dumps(r['distance_to_fixed_goal_at_generated_window_ends_cm']),boundary_velocity_discontinuity_m_s=json.dumps(r['boundary_velocity_discontinuity_m_s'])) for r in allrows])
    csvwrite(OUT/'ENDPOINT_SUMMARY.csv',summary);csvwrite(OUT/'PAIRED_MODEL_CONTRASTS.csv',contrasts)
    write(OUT/'ENDPOINT_AUDIT.json',dict(status='PASS_ALL_14168_ENDPOINTS_REAL_GOALS_AND_NO_ROUTE_PENALTY',checks=audit,time=time.time()))

def quality_task(task):
    banks={m:bank(m,task) for m in MODELS};ref=banks[MODELS[0]];dest=OUT/task
    for m,b in banks.items():
        for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(b[k],ref[k]),(task,m,k)
        for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:assert np.allclose(b[k],ref[k],atol=1e-4,rtol=1e-5),(m,k)
    # Candidate pools are fixed by source IDs, so a changed deadline does not
    # silently change semantic retrieval difficulty or negative examples.
    selected=[];seen=set()
    for i in sorted(range(506),key=lambda i:str(ref['source_ids'][i])):
        caption=str(ref['captions'][i])
        if caption not in seen:seen.add(caption);selected.append(i)
    q=np.array(selected);records=[dict(sample_id=str(ref['source_ids'][i]),source_family_id=str(ref['source_ids'][i]),caption=str(ref['captions'][i]),frames=52) for i in q]
    pools={str(s):candidate_pools(records,s,length_matched=True) for s in [4101,4102,4103]}
    write(dest/'QUERY_POOLS.json',dict(query_indices=selected,query_source_ids=[r['sample_id'] for r in records],pools=pools,scope='same source-keyed deduplicated-caption32-way pools across all models and deadlines'))
    frows=[];rrows=[]
    for w,start in enumerate(ref['starts']):
        cache=dest/f'quality_w{w+1:02d}.json'
        if cache.exists():
            old=read(cache);frows+=old['FID'];rrows+=old['retrieval'];continue
        real=ref['distribution_real'][w];generated=[banks[m]['distribution_generated'][w] for m in MODELS]
        point=fid_to_many(real,generated);rng=np.random.default_rng(20261017+w)
        boot=np.array([fid_to_many(real[ix],[g[ix] for g in generated]) for ix in [cluster_draw(ref['actor_ids'],rng) for _ in range(500)]])
        fr=[]
        for j,m in enumerate(MODELS):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);fr.append(dict(task=task,model=m,window=w+1,start_generated_frame=int(start),end_generated_frame_exclusive=int(start+52),N_sources=506,LocalFID52=float(point[j]),FID_ci95_low=float(lo),FID_ci95_high=float(hi),bootstrap_repeats=500,scope='conditional natural-reference local52 distribution; not full-sequence FID; not far27 stratum FID'))
        rr=[];qactors=ref['actor_ids'][q];rng=np.random.default_rng(3101701+w);draws=[cluster_draw(qactors,rng) for _ in range(2000)]
        for seed in [1701,2903]:
            tt=ref[f'text_seed{seed}'][w,q]
            real_ranks=np.array([retrieval(tt,ref[f'retrieval_real_seed{seed}'][w,q],p)[1] for p in pools.values()])
            for m in MODELS:
                ranks=np.array([retrieval(tt,banks[m][f'retrieval_generated_seed{seed}'][w,q],p)[1] for p in pools.values()])
                r=dict(task=task,model=m,window=w+1,encoder_seed=seed,N_queries=len(q),candidates=32,source_pool_seeds='4101/4102/4103',scope='local active-instruction compatibility, not event-order accuracy')
                for k in [1,2,3]:
                    success=(ranks<=k).mean(axis=0);bs=np.array([success[ix].mean() for ix in draws]);lo,hi=np.quantile(bs,[.025,.975])
                    r.update({f'R{k}':float(success.mean()),f'R{k}_ci95_low':float(lo),f'R{k}_ci95_high':float(hi),f'real_R{k}':float((real_ranks<=k).mean())})
                rr.append(r)
        write(cache,dict(FID=fr,retrieval=rr,window=w+1,time=time.time()));frows+=fr;rrows+=rr
        print('QUALITY_WINDOW',task,w+1,len(ref['starts']),flush=True)
    csvwrite(dest/'WINDOW_FID.csv',frows);csvwrite(dest/'WINDOW_RETRIEVAL.csv',rrows)
    write(dest/'COMPLETE.json',dict(status='COMPLETE_PAIRED_LOCAL52_QUALITY',N=506,windows=len(ref['starts']),models=MODELS,N_queries=len(q),time=time.time()))

def main():
    endpoints()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(quality_task,TASKS))
    pools=[read(OUT/t/'QUERY_POOLS.json') for t in TASKS]
    assert all(p['query_source_ids']==pools[0]['query_source_ids'] and p['pools']==pools[0]['pools'] for p in pools),'Deadline changed retrieval pool'
    f=[];r=[]
    for task in TASKS:
        f+=list(csv.DictReader((OUT/task/'WINDOW_FID.csv').open()));r+=list(csv.DictReader((OUT/task/'WINDOW_RETRIEVAL.csv').open()))
    csvwrite(OUT/'ALL_WINDOW_LOCALFID52.csv',f);csvwrite(OUT/'ALL_WINDOW_RETRIEVAL.csv',r)
    write(ROOT/'NUMERICS_COMPLETE.json',dict(status='COMPLETE_ALL_FOUR_MODELS_CONTROLLED_ENDPOINTS_AND_FROZEN_LOCAL52_METRICS',new_trajectories=14168,N_sources=506,actors=22,models=MODELS,tasks=TASKS,
        endpoint_audit=sha(OUT/'ENDPOINT_AUDIT.json'),result_hashes={n:sha(OUT/n) for n in ['ENDPOINT_SUMMARY.csv','PAIRED_MODEL_CONTRASTS.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']},
        source_exposure='generator-training-exposed post-hoc controlled diagnosis, not independent unseen benchmark',no_root_ADE=True,time=time.time()))
    print('NUMERICS_COMPLETE',flush=True)

if __name__=='__main__':main()
