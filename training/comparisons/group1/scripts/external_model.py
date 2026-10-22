"""Disclosed common-G1-codec architecture controls, not official systems.

Official DiP decoder / DART residual MLP forward math is executed from pinned
source. External physical/text assets are replaced by the common frozen G1
codec and cached4096D text. Conditioning is wired through input dimensions,
without a Mamba, planner, learned memory or auxiliary control network.
"""
from pathlib import Path
import ast, hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MAX_FRAMES=248
class FP32Linear(nn.Linear):
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return F.linear(x.float(),self.weight,self.bias)

def official_classes(path,method):
    tree=ast.parse(Path(path).read_text());classes=[]
    edits=[]
    for n in tree.body:
        if not isinstance(n,ast.ClassDef):continue
        if method=='dip' and n.name=='MDM':
            n.body=[f for f in n.body if not isinstance(f,ast.FunctionDef) or f.name not in ['_apply','train']]
            init=next(f for f in n.body if isinstance(f,ast.FunctionDef) and f.name=='__init__')
            init.body=[f for f in init.body if not (isinstance(f,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='rot2xyz' for t in f.targets))]
            edits=['omit unused SMPL Rotation2xyz constructor','inherit standard nn.Module train/to; no SMPL side effects']
        classes.append(n)
    env=dict(torch=torch,nn=nn,np=np,F=F,lora=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(path),'exec'),env)
    return env,edits

class ExternalDenoiser(nn.Module):
    def __init__(self,method,repo,motion_rep,hybrid):
        super().__init__();self.method=method
        object.__setattr__(self,'motion_rep',motion_rep)
        object.__setattr__(self,'hybrid',hybrid)
        self.nframe_root_dim=20
        self.training_precision_contract='g1_root_output_geometry_fp32_v1'
        source=Path(repo)/('model/mdm.py' if method=='dip' else 'model/mld_denoiser.py')
        env,self.source_edits=official_classes(source,method)
        self.source=str(source);self.source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
        if method=='dip':
            Base=env['MDM']
            class CachedTextMDM(Base):
                def load_and_freeze_clip(self,version):return nn.Identity()
            self.core=CachedTextMDM(modeltype='',njoints=148+4*828+5,nfeats=1,num_actions=1,
                translation=True,pose_rep='rot6d',glob=True,glob_rot=True,latent_dim=512,
                ff_size=1024,num_layers=8,num_heads=4,dropout=.1,activation='gelu',
                data_rep='hml_vec',dataset='g1_shared',clip_dim=4096,cond_mode='text',
                cond_mask_prob=0.,arch='trans_dec',text_encoder_type='clip',mask_frames=True,
                pred_len=0,context_len=0,emb_policy='add')
            # Common representations require a different input and output width.
            self.core.output_process=env['OutputProcess']('hml_vec',148,512,148,1)
            final=self.core.output_process.poseFinal
            replacement=FP32Linear(final.in_features,final.out_features)
            replacement.load_state_dict(final.state_dict());self.core.output_process.poseFinal=replacement
        else:
            self.core=env['DenoiserMLP'](h_dim=512,n_blocks=2,dropout=.1,activation='gelu',
                clip_dim=4096+MAX_FRAMES*828+5,history_shape=(49,148),noise_shape=(13,148),cond_mask_prob=0.)
            final=self.core.mlp.out_fc;replacement=FP32Linear(final.in_features,final.out_features)
            replacement.load_state_dict(final.state_dict());self.core.mlp.out_fc=replacement

    def inputs(self,x,history_len,generation_len,future_len,history_mask,generation_mask,future_mask,
               history_token_mask,generation_token_mask,future_token_mask,text_feat,text_pad_mask,t,
               first_heading_angle,motion_mask=None,observed_motion=None):
        b,n,d=x.shape;nf=n*4
        assert d==148 and nf<=MAX_FRAMES
        assert bool((generation_len==52).all())
        assert torch.equal(generation_token_mask,generation_token_mask[:1].expand_as(generation_token_mask))
        mask=motion_mask.float();obs=observed_motion.float()*mask
        text=text_feat.float()[:,0]*text_pad_mask[:,0,None]
        layout=torch.stack([history_len,generation_len,future_len],-1).float()/MAX_FRAMES
        angle=torch.stack([torch.sin(first_heading_angle),torch.cos(first_heading_angle)],-1)
        roles=torch.stack([history_token_mask,generation_token_mask,future_token_mask],-1).float()
        if self.method=='dip':
            cond=torch.cat([obs,mask],-1).reshape(b,n,4*828)
            a=torch.cat([x.float(),cond,roles,angle[:,None].expand(-1,n,-1)],-1)
            y=dict(mask=(history_token_mask|generation_token_mask|future_token_mask)[:,None,None,:],
                   text_embed=text[None],uncond=False)
            return a.transpose(1,2)[:,:,None],t,y
        hist=x.new_zeros(b,49,148)
        h=int(history_len[0]//4);assert bool((history_len==history_len[0]).all())
        if h:hist[:,-h:]=x[:,:h]
        cond=F.pad(torch.cat([obs,mask],-1),(0,0,0,MAX_FRAMES-nf)).flatten(1)
        y=dict(history_motion_normalized=hist,text_embedding=torch.cat([text,cond,layout,angle],-1),uncond=False)
        return x[:,generation_token_mask[0]],t,y

    def forward(self,*args,**kwargs):
        bound=dict(zip(['x','history_len','generation_len','future_len','history_mask','generation_mask','future_mask',
            'history_token_mask','generation_token_mask','future_token_mask','text_feat','text_pad_mask','t',
            'first_heading_angle','motion_mask','observed_motion'],args));bound.update(kwargs)
        a,t,y=self.inputs(**bound);out=self.core(a,t,y)
        if self.method=='dip':return out[:,:,0].transpose(1,2)
        full=torch.zeros_like(bound['x']);full[:,bound['generation_token_mask'][0]]=out
        return full
