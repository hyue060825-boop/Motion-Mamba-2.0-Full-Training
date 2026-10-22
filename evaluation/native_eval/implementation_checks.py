"""Bounded implementation diagnostics, not evaluator quality qualification."""
import io
import unittest
import numpy as np
import torch
from .common import *

def unit_checks():
    suite=unittest.defaultTestLoader.discover(str(CODE/'tests'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    out=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),log=stream.getvalue(),code=snapshot_code())
    write(ROOT/'IMPLEMENTATION_TESTS.json',out)
    if not result.wasSuccessful():raise RuntimeError('Implementation checks failed')
    return out

def overfit_check():
    p=ROOT/'SMALL_SUBSET_LEARNING_CHECK.json'
    if p.exists():return read(p)
    from .text import TextCache
    from .train import collate,MotionDataset,seed_all
    from .model import DualEncoder,contrastive,reconstruction_loss
    cfg=read(CODE/'config.json');seed_all(1717);torch.set_num_threads(2)
    text=TextCache();rr=[];captions=set()
    for r in rows(ROOT/'data/bones_train.jsonl'):
        if r['frames']<=104 and r['captions'][0] not in captions:
            rr.append(dict(r,caption=r['captions'][0]));captions.add(r['captions'][0])
        if len(rr)==32:break
    stats=dict(np.load(ROOT/'data/bones_stats.npz'));ds=MotionDataset(rr,text,stats)
    x,l,t,records=collate([ds[i] for i in range(len(ds))]);model=DualEncoder(cfg)
    mean=torch.tensor(stats['mean']);std=torch.tensor(stats['std'])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4);losses=[];started=time.time()
    for step in range(80):
        model.train();opt.zero_grad(set_to_none=True);h,m=model.motion(x,l);tt=model.text(t)
        loss=contrastive(m,tt,t,records,.1)+reconstruction_loss(model.reconstruct(h,l,x.shape[1]),x,l,mean,std)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    out=dict(device='CPU',updates=80,bones_gpu_pilot_updates=120,total_pilot_learning_updates=200,
       source_ids=[r['sample_id'] for r in rr],initial_loss=float(np.mean(losses[:10])),final_loss=float(np.mean(losses[-10:])),
       learning_observed=bool(np.mean(losses[-10:])<.7*np.mean(losses[:10])),seconds=time.time()-started,
       limitation='same tiny train subset only; no claim of generalization or release validity',losses=losses)
    write(p,out);ledger('small_subset_learning',**{k:v for k,v in out.items() if k!='losses'});return out

def near_duplicate_review():
    p=ROOT/'NEAR_DUPLICATE_REVIEW.json'
    if p.exists():return read(p)
    from scipy.spatial import cKDTree
    results={}
    for d in ['bones','humanml3d']:
        train=sorted(rows(ROOT/'data'/f'{d}_train.jsonl'),key=lambda r:identity(['near-dup',r['sample_id']]))[:512]
        dev=sorted(rows(ROOT/'data'/f'{d}_development.jsonl'),key=lambda r:identity(['near-dup',r['sample_id']]))[:128]
        def descriptor(r):
            a=np.load(r['feature_path'],mmap_mode='r')
            pos=a[:,5:104];idx=np.linspace(0,len(pos)-1,8).astype(int)
            return pos[idx].reshape(-1)
        a=np.stack([descriptor(r) for r in train]);b=np.stack([descriptor(r) for r in dev]);distance,ix=cKDTree(a).query(b,workers=1)
        closest=np.argsort(distance)[:8]
        results[d]=dict(train_sampled=len(a),development_sampled=len(b),method='8 normalized-time physical-position frames, Euclidean nearest neighbour; exploratory only',
           closest=[dict(development_id=dev[i]['sample_id'],train_id=train[ix[i]]['sample_id'],position_rmse_m=float(distance[i]/np.sqrt(a.shape[1])),
                    same_family=dev[i]['source_family_id']==train[ix[i]]['source_family_id'],same_source_hash=dev[i]['source_hashes'].get('motion')==train[ix[i]]['source_hashes'].get('motion'),
                    captions=[dev[i]['captions'][0],train[ix[i]]['captions'][0]]) for i in closest],
           conclusion='Source/family identity reviewed for closest sampled pairs. Similar static or repeated semantic actions are not automatically duplicates. This sample does not exclude all unknown cross-source retargeted near-duplicates.')
    write(p,results);return results
