"""Release inference is FP32 with fixed length-sorted batch32, restored by ID."""
import numpy as np
import torch
from .schema import canonicalize

@torch.inference_mode()
def encode_arrays(model,records,arrays,stats,texts=None):
    device=next(model.parameters()).device;model.eval();lookup={};lookup_text={}
    order=sorted(range(len(records)),key=lambda i:(len(arrays[i]),records[i]['sample_id']))
    for start in range(0,len(order),32):
        batch=order[start:start+32];lengths=torch.tensor([len(arrays[i]) for i in batch],device=device)
        x=torch.zeros(len(batch),int(lengths.max()),414,device=device)
        for j,i in enumerate(batch):x[j,:len(arrays[i])]=torch.tensor((canonicalize(arrays[i])-stats['mean'])/stats['std'],device=device)
        h,m=model.motion(x,lengths)
        if texts is not None:t=model.text(torch.tensor(np.stack([texts[i] for i in batch]),device=device))
        for j,i in enumerate(batch):
            lookup[records[i]['sample_id']]=(h[j].cpu().numpy(),m[j].cpu().numpy())
            if texts is not None:lookup_text[records[i]['sample_id']]=t[j].cpu().numpy()
    ids=[r['sample_id'] for r in records]
    return np.stack([lookup[k][0] for k in ids]),np.stack([lookup[k][1] for k in ids]),np.stack([lookup_text[k] for k in ids]) if texts is not None else None

def encode(model,records,text,stats,mode='full',perturb=None):
    arrays=[]
    for r in records:
        a=np.array(np.load(r['feature_path'],mmap_mode='r'),dtype='float32')
        if mode=='52':
            s=(len(a)-52)//2;a=a[s:s+52]
        if perturb:a=perturb(a,r)
        arrays.append(a)
    return encode_arrays(model,records,arrays,stats,[text[r['caption']] for r in records])
