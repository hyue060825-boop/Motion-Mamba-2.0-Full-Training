import math
import torch
from torch import nn
from torch.nn import functional as F

def positional(t,dim,device):
    scales=torch.exp(torch.arange(0,dim,2,device=device)*(-math.log(10000.)/dim))
    phase=torch.arange(t,device=device)[:,None]*scales
    return torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)

class DualEncoder(nn.Module):
    def __init__(self,cfg,text_dim=1024,variant=None):
        super().__init__();h=cfg['hidden'];self.hidden=h;self.variant=variant or cfg.get('variant','R0')
        self.input=nn.Linear(414,h)
        self.ordered_input=nn.Linear(414*4,h) if self.variant=='R2' else None
        block=nn.TransformerEncoderLayer(h,cfg['heads'],h*4,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,cfg['layers'],enable_nested_tensor=False)
        self.final=nn.LayerNorm(h)
        self.motion_projection=nn.Linear(h,h)
        self.text_projection=nn.Sequential(nn.Linear(text_dim,h),nn.GELU(),nn.Linear(h,h))
        self.decoder=nn.Sequential(nn.Linear(h+34,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,414))

    def motion(self,x,lengths):
        # Four-frame pooling is encoder-only; no generator codec or frozen motion weights.
        valid=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        pad=(-x.shape[1])%4
        v=F.pad(valid,(0,pad)).reshape(len(x),-1,4)
        if self.variant=='R2':
            z=F.pad(x.masked_fill(~valid[:,:,None],0),(0,0,0,pad)).reshape(len(x),-1,4,414)
            y=self.ordered_input(z.reshape(len(x),-1,1656))
        else:
            y=self.input(x)*valid[:,:,None]
            y=F.pad(y,(0,0,0,pad)).reshape(len(x),-1,4,self.hidden).sum(2)/v.sum(2).clamp_min(1)[:,:,None]
        y=y+positional(y.shape[1],self.hidden,x.device)[None]
        mask=~v.any(2)
        y=self.encoder(y,src_key_padding_mask=mask)
        h=self.final((y*(~mask)[:,:,None]).sum(1)/(~mask).sum(1)[:,None])
        return h,F.normalize(self.motion_projection(h),dim=-1)

    def text(self,t):return F.normalize(self.text_projection(t),dim=-1)

    def reconstruct(self,h,lengths,frames):
        pos=positional(frames,32,h.device)[None].expand(len(h),-1,-1)
        q=torch.arange(frames,device=h.device)[None].expand(len(h),-1)/lengths[:,None]
        length=(lengths.float()/500)[:,None,None].expand(-1,frames,1)
        return self.decoder(torch.cat((h[:,None].expand(-1,frames,-1),pos,q[:,:,None],length),-1))

def contrastive(motion,text,text_raw,records,temperature):
    logits=motion@text.T/temperature;b=len(records)
    positive=torch.tensor([[r['sample_id']==s['sample_id'] for s in records] for r in records],device=logits.device)
    ambiguity=torch.tensor([[r['source_family_id']==s['source_family_id'] or r['caption']==s['caption'] for s in records] for r in records],device=logits.device)
    raw=F.normalize(text_raw.float(),dim=-1)
    ambiguity=ambiguity|((raw@raw.T)>.95)
    allowed=~ambiguity|positive
    def one(z,p,a):
        logp=z.masked_fill(~a,-1e9).log_softmax(-1)
        return -((logp*p).sum(-1)/p.sum(-1)).mean()
    return (one(logits,positive,allowed)+one(logits.T,positive.T,allowed.T))/2

def reconstruction_loss(pred,target,lengths,mean,std):
    valid=torch.arange(target.shape[1],device=target.device)[None,:]<lengths[:,None]
    losses=[]
    for start,end in [(0,3),(3,5),(5,104),(104,308),(308,410)]:
        loss=((pred[:,:,start:end]-target[:,:,start:end])**2).mean(-1)
        losses.append(loss[valid].mean())
    contact=(target[:,:,410:]*std[410:]+mean[410:]).clamp(0,1)
    losses.append(F.binary_cross_entropy_with_logits(pred[:,:,410:][valid],contact[valid]))
    raw=pred[:,:,104:308]*std[104:308]+mean[104:308]
    rot=raw.reshape(*raw.shape[:2],34,6);a,b=rot[...,:3],rot[...,3:]
    regular=((a.norm(dim=-1)-1)**2+(b.norm(dim=-1)-1)**2+(a*b).sum(-1)**2).mean(-1)[valid].mean()
    return torch.stack(losses).mean()+.01*regular
