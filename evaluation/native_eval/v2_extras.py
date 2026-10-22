"""Read-only v1 replay and covariance-change calibration; CPU only."""
from motion_mamba2.paths import artifact_path
import importlib.util
import numpy as np
import torch
from .common import *
from .metrics import fid,moments,psd_sqrt
from .v2_statistics import interval

def scale_delta(a,b,scale):
    ma,ca=moments(a);mb,cb=moments(b);root=psd_sqrt(ca);mid=root@cb@root
    cross=np.sqrt(np.linalg.eigvalsh((mid+mid.T)/2).clip(0)).sum()
    return float(((ma-scale*mb)**2).sum()-((ma-mb)**2).sum()+(scale*scale-1)*np.trace(cb)-2*(scale-1)*cross)

def covariance_calibration():
    target=ROOT/'benchmarks/COVARIANCE_CALIBRATION.json'
    if target.exists():return read(target)
    rng=np.random.default_rng(61104);out=[]
    for dim,n in ((16,112),(256,112),(256,512)):
        records=[];alpha=.8;truth=dim*(1-alpha)**2
        for trial in range(24):
            a=rng.normal(size=(n,dim));b=rng.normal(size=(n,dim));point=scale_delta(a,b,alpha)
            if trial==0 and abs(point-(fid(a,alpha*b)-fid(a,b)))>1e-4:raise ValueError('Scale equivalence mismatch')
            boots=[scale_delta(a[rng.integers(n,size=n)],b[rng.integers(n,size=n)],alpha) for _ in range(500)]
            records.append(dict(trial=trial,truth=truth,**interval(point,boots)))
        out.append(dict(dimension=dim,per_side_N=n,scale=alpha,bootstrap_repeats=500,trials=24,population_delta=truth,basic_coverage=sum(r['basic_ci95'][0]<=truth<=r['basic_ci95'][1] for r in records)/24,percentile_coverage=sum(r['percentile_ci95'][0]<=truth<=r['percentile_ci95'][1] for r in records)/24,finite_sample_delta_bias=float(np.mean([r['point'] for r in records])-truth),records=records))
        write(ROOT/'benchmarks/COVARIANCE_CALIBRATION_PROGRESS.json',out)
    result=dict(results=out,interpretation='Separate covariance-scale experiment; paired delta can itself have finite-N bias. Do not claim population-calibrated intervals from location-shift success.',trial_limit='24 Monte Carlo trials: coverage estimate noisy; failures remain evidence, success not a universal certificate')
    write(target,result);return result

def replay():
    target=ROOT/'benchmarks/V1_STATISTICAL_REPLAY.json'
    if target.exists():return read(target)
    v1=Path(artifact_path('evaluator_native_67895f3cd5'));out={};cfg=read(artifact_path('config_json_8d4866da1a'))
    spec=importlib.util.spec_from_file_location('v1_matched_model',artifact_path('model_py_916e1b8896'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for d in ('humanml3d','bones'):
        a=np.load(v1/'validation'/d/'seed1701/confirmation/embeddings.npz')['h52'];curves=[]
        for n in (112,256,512):
            if 2*n>len(a):continue
            values=[]
            for seed in (1004,2004,3004,4004,5004):
                ix=np.random.default_rng(seed).permutation(len(a));values.append(fid(a[ix[:n]],a[ix[n:2*n]]))
            curves.append(dict(each_side_N=n,real_real=values))
        ck=torch.load(v1/'runs'/d/'seed1701/best.pt',map_location='cpu',weights_only=False)
        if ck['contract']['architecture_code_sha256']!=sha(artifact_path('model_py_916e1b8896')):raise ValueError('v1 architecture not matching checkpoint')
        model=mod.DualEncoder(cfg).eval();model.load_state_dict(ck['model']);stats=np.load(v1/'data'/f'{d}_stats.npz');rr=read(v1/'panels'/f'{d}_confirmation.json')['records'][:16];diff=[]
        for r in rr:
            x=np.load(r['feature_path']);start=(len(x)-52)//2;x=x[start:start+52].copy();x[:,[0,2]]-=x[0,[0,2]];x=(x-stats['mean'])/stats['std'];y=x.copy()
            for s in range(0,52,4):y[s+1],y[s+2]=x[s+2].copy(),x[s+1].copy()
            with torch.no_grad():h=model.motion(torch.tensor(np.stack([x,y])),torch.tensor([52,52]))[0]
            diff.append(float((h[0]-h[1]).abs().max()))
        out[d]=dict(N=len(a),curves=curves,within_block_permutation_max_abs=max(diff),v1_weights_sha256=sha(v1/'runs'/d/'seed1701/best.pt'),diagnostic='complete feature-row block permutation; no velocity reconstruction; tests structure information loss only')
    write(target,out);return out

if __name__=='__main__':
    torch.set_num_threads(2);replay();covariance_calibration();print('CPU replay and covariance calibration complete')
