"""Standard FID; explicitly limited finite-sample uncertainty diagnostics."""
import time
import numpy as np
from .common import *
from .metrics import fid,moments,psd_sqrt,gaussian_fid,embedding_health

def health(x):
    result=embedding_health(x);result['covariance_rank']=int(np.linalg.matrix_rank(np.cov(np.asarray(x,np.float64),rowvar=False)))
    return result

def interval(point,boot):
    q=np.quantile(boot,[.025,.975]);return dict(point=float(point),percentile_ci95=q.tolist(),basic_ci95=[float(2*point-q[1]),float(2*point-q[0])],bootstrap_mean=float(np.mean(boot)),estimated_bias=float(np.mean(boot)-point))

def draw_groups(indices,groups,rng):
    labels=np.asarray(groups)[indices];unique=sorted(set(labels.tolist()))
    members=[indices[labels==g] for g in unique]
    return np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])

def fid_to_many(reference,candidates):
    mu,c= moments(reference);root=psd_sqrt(c);out=[]
    for other in candidates:
        om,oc=moments(other);middle=root@oc@root
        eigen=np.linalg.eigvalsh((middle+middle.T)/2)
        if eigen.min() < -1e-8*max(1.,np.abs(eigen).max()):raise ValueError('non-PSD covariance')
        f=float(((mu-om)**2).sum()+np.trace(c)+np.trace(oc)-2*np.sqrt(eigen.clip(0)).sum())
        out.append(max(0.,f))
    return np.array(out)

def distribution(h,variants,records,repeats=1000,actor_cluster=False):
    names=list(variants);all_features=[h]+[variants[k] for k in names]
    groups=[r['actor_id_namespaced'] if actor_cluster else r['source_family_id'] for r in records]
    unique=sorted(set(groups));split_summaries=[]
    for seed in (1004,2004,3004,4004,5004):
        rng=np.random.default_rng(seed)
        if actor_cluster:
            ug=np.asarray(unique);ga,gb=np.array_split(ug[rng.permutation(len(ug))],2)
            a=np.flatnonzero(np.isin(groups,ga));b=np.flatnonzero(np.isin(groups,gb))
        else:
            a,b=np.array_split(rng.permutation(len(h)),2);n=min(len(a),len(b));a,b=a[:n],b[:n]
        values=fid_to_many(h[a],[v[b] for v in all_features])
        split_summaries.append(dict(seed=seed,a=a,b=b,N_a=len(a),N_b=len(b),clusters_a=len(set(np.asarray(groups)[a])),clusters_b=len(set(np.asarray(groups)[b])),real_real=float(values[0]),deltas={k:float(values[i+1]-values[0]) for i,k in enumerate(names)}))
    first=split_summaries[0];a,b=first['a'],first['b'];rng=np.random.default_rng(77004);boots=[];t=time.time()
    for _ in range(repeats):
        aa=draw_groups(a,groups,rng);bb=draw_groups(b,groups,rng)
        values=fid_to_many(h[aa],[v[bb] for v in all_features]);boots.append(np.r_[values[0],values[1:]-values[0]])
    boots=np.array(boots);result={}
    for i,k in enumerate(names):
        result[k]=interval(first['deltas'][k],boots[:,i+1]);result[k].update(paired_feature_rms=float(np.sqrt(np.mean((h-variants[k])**2))),paired_self_reference_fid=fid(h,variants[k]),positive_splits=sum(s['deltas'][k]>0 for s in split_summaries))
    rr=interval(first['real_real'],boots[:,0]);rr['warning']='raw bootstrap FID is upward biased in high dimension; neither interval is assumed population-calibrated'
    curve=[]
    for n in (112,256,512):
        if 2*n>len(h):continue
        values=[]
        for seed in (1004,2004,3004,4004,5004):
            ii=np.random.default_rng(seed).permutation(len(h))[:2*n];values.append(fid(h[ii[:n]],h[ii[n:]]))
        curve.append(dict(per_side_N=n,values=values,mean=float(np.mean(values)),std=float(np.std(values))))
    return dict(N=len(h),families=len({r['source_family_id'] for r in records}),actors=len({r['actor_id_namespaced'] for r in records}),health=health(h),side_health=dict(a=health(h[a]),b=health(h[b])),cluster_unit='actor' if actor_cluster else 'family',real_real=rr,splits=[{k:v for k,v in s.items() if k not in ('a','b')} for s in split_summaries],sample_size_curve=curve,variants=result,bootstrap_repeats=repeats,seconds=time.time()-t)

def calibrate():
    target=ROOT/'benchmarks/STATISTICAL_CALIBRATION.json'
    if target.exists():return read(target)
    rng=np.random.default_rng(51004);results=[];started=time.time()
    # Location-shift subcase has an exact cancellation of covariance terms;
    # use this algebraic equivalence for Monte Carlo, verify against full FID.
    # Covariance-scale case is separately diagnosed, never assumed calibrated.
    for dim,n in ((16,112),(256,112),(256,512)):
        trials=[];rawbias=[]
        for trial in range(100):
            a=rng.normal(size=(n,dim));b=rng.normal(size=(n,dim));shift=np.full(dim,.1);point=float(2*((b.mean(0)-a.mean(0))*shift).sum()+shift@shift)
            boot=[]
            for _ in range(500):
                aa=a[rng.integers(n,size=n)];bb=b[rng.integers(n,size=n)]
                boot.append(2*((bb.mean(0)-aa.mean(0))*shift).sum()+shift@shift)
            ci=interval(point,boot);truth=float(shift@shift);trials.append(dict(point=point,truth=truth,basic=ci['basic_ci95'],percentile=ci['percentile_ci95']))
            if trial<5:
                base=fid(a,b);equiv=fid(a,b+shift)-base
                if abs(point-equiv)>1e-4:raise ValueError('Gaussian shift equivalence failed')
                raw=[]
                for _ in range(100):raw.append(fid(a[rng.integers(n,size=n)],b[rng.integers(n,size=n)]))
                rawbias.append(dict(real_real=base,bootstrap_mean=float(np.mean(raw)),gaussian_population_fid=0.,bias=float(np.mean(raw)-base)))
        coverage=sum(v['basic'][0]<=v['truth']<=v['basic'][1] for v in trials)/len(trials)
        # Null paired edit returns exactly the same covariance/mean; no false
        # effect possible. A stochastic zero-mean signed shift adds a second
        # centered-null check with known expected paired effect removed.
        false_positive=sum(v['basic'][0]-v['truth']>0 or v['basic'][1]-v['truth']<0 for v in trials)/len(trials)
        results.append(dict(dimension=dim,per_side_N=n,trials=100,bootstrap_repeats=500,location_shift_coverage=coverage,centered_null_false_positive=false_positive,raw_fid_bootstrap_bias=rawbias,limitation='location shifts only; does not certify covariance-changing motion edits'))
    result=dict(results=results,seconds=time.time()-started,raw_fid_population_ci_calibrated=False,paired_location_delta_calibrated=all(r['location_shift_coverage']>=.85 for r in results),covariance_edit_population_coverage='NOT_ESTABLISHED; paired finite-panel intervals are descriptive, robust sign and large-effects required; no nominal population-coverage claim')
    write(target,result);return result
