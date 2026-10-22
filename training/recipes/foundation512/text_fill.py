from common import *
import numpy as np, torch
from motion_training.data import text_key
missing=read(ROOT/'TEXT_MISSING.json')
if missing:
    from motion_mamba2.models.load_model import load_text_encoder
    torch.set_num_threads(2);encoder=load_text_encoder(mode='local',device='cuda:0')
    with torch.inference_mode():
        for i,text in enumerate(missing):
            p=ROOT/'text'/(text_key(text)+'.npy')
            if p.exists():continue
            v,n=encoder(text);v=v[:n].float().cpu().numpy()
            assert v.size==4096 and np.isfinite(v).all()
            tmp=p.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(p)
            if i%100==0:print('OFFICIAL_TEXT',i+1,len(missing),flush=True)
write(ROOT/'TEXT_FILL_COMPLETE.json',dict(encoded=len(missing),exact_official_encoder=True))
