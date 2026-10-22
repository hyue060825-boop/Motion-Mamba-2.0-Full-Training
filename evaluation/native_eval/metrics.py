"""Float64 global FID; text (rows) -> motion (columns) retrieval."""
import numpy as np

def moments(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all(): raise ValueError('Need >=2 finite feature rows')
    return x.mean(0),np.cov(x,rowvar=False,ddof=1)

def psd_sqrt(c):
    c=(c+c.T)/2
    w,v=np.linalg.eigh(c)
    tol=1e-8*max(1.,float(np.max(np.abs(w))))
    if w.min() < -tol: raise ValueError('Covariance is not PSD within frozen tolerance')
    return (v*np.sqrt(w.clip(0)))@v.T

def gaussian_fid(ma,ca,mb,cb):
    ma,ca,mb,cb=[np.asarray(v,np.float64) for v in [ma,ca,mb,cb]]
    a=psd_sqrt(ca);middle=a@cb@a
    val=float(np.sum((ma-mb)**2)+np.trace(ca)+np.trace(cb)-2*np.trace(psd_sqrt(middle)))
    tol=1e-6*max(1.,float(np.trace(ca)+np.trace(cb)))
    if val < -tol:raise ValueError('Negative FID beyond frozen numerical tolerance')
    return max(0.,val)

def fid(a,b): return gaussian_fid(*moments(a),*moments(b))

def normalize(x):
    x=np.asarray(x,np.float64);n=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or (n<1e-12).any():raise ValueError('Invalid/zero retrieval embedding')
    return x/n

def distances(text,motion):
    return np.maximum(0,2-2*normalize(text)@normalize(motion).T)

def ranks(distance,candidates,positive):
    distance=np.asarray(distance)
    result=[]
    for i,(ids,target) in enumerate(zip(candidates,positive)):
        if len(ids)!=len(set(ids)) or target not in ids:raise ValueError('Ambiguous candidate identities')
        # Candidate IDs sorted by frozen sample ID; stable sorting resolves ties.
        order=np.argsort(distance[i,ids],kind='stable')
        result.append(1+list(np.asarray(ids)[order]).index(target))
    return np.asarray(result)

def retrieval(text,motion,pools):
    if any(len(x)!=32 for x in pools):raise ValueError('Strict native R requires exactly 32 candidates')
    rank=ranks(distances(text,motion),pools,list(range(len(text))))
    return {f'R@{k}':float(np.mean(rank<=k)) for k in [1,2,3]},rank

def candidate_pools(records,seed,length_matched=False):
    if len(records)<32:raise ValueError('Insufficient candidates: need 32')
    rng=np.random.default_rng(seed); pools=[]
    for i,r in enumerate(records):
        allowed=[j for j,s in enumerate(records) if j!=i and s['source_family_id']!=r['source_family_id'] and s['caption']!=r['caption']]
        if length_matched:
            allowed=[j for j in allowed if abs(records[j]['frames']-r['frames'])<=max(8,round(r['frames']*.25))]
        if len(allowed)<31:raise ValueError('Insufficient distinct family/caption length-matched candidates')
        chosen=rng.choice(allowed,31,replace=False).tolist()+[i]
        pools.append(sorted(chosen,key=lambda j:records[j]['sample_id']))
    return pools

def family_mean_ci(values,families,seed=1004,repeats=200):
    # Unequal family groups are kept intact, rather than bootstrapping windows independently.
    values=np.asarray(values); unique=sorted(set(families));groups=[np.flatnonzero(np.asarray(families)==f) for f in unique]
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repeats):
        ids=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]);samples.append(float(values[ids].mean()))
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),families=len(unique),repeats=repeats)

def embedding_health(h):
    h=np.asarray(h,np.float64);sv=np.linalg.svd(h-h.mean(0),compute_uv=False)**2
    p=sv/sv.sum() if sv.sum()>0 else np.zeros_like(sv)
    return dict(mean_variance=float(h.var(0).mean()),norm_mean=float(np.linalg.norm(h,axis=1).mean()),
                effective_rank=float(np.exp(-(p*np.log(p.clip(1e-30))).sum())),samples=len(h),dimension=h.shape[1])

def local_starts(frames,width=52,stride=52):
    if frames<width:raise ValueError('Short trajectory')
    starts=list(range(0,frames-width+1,stride))
    if starts[-1]!=frames-width:starts.append(frames-width)
    return starts
