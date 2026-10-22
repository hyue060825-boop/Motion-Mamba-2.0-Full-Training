"""Official MotionMamba core with disclosed common-G1 representation adapters."""
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import ast,math,copy,hashlib,typing,sys
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from einops import rearrange,repeat

def definitions(path,env,names=None):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (names is None or n.name in names)]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
 return env

class FP32Linear(nn.Linear):
 def forward(self,x):
  with torch.autocast(x.device.type,enabled=False):return F.linear(x.float(),self.weight,self.bias)

class BatchFirstScan(nn.Module):
 """DFMBlock uses [T,B,C]; official Mamba scan kernels require [B,T,C]."""
 def __init__(self,operation):super().__init__();self.operation=operation
 def forward(self,x,**kw):return self.operation(x.transpose(0,1).contiguous(),**kw).transpose(0,1).contiguous()

def official_environment(repo):
 from mamba_ssm import Mamba,selective_scan_fn,mamba_inner_fn
 from mamba_ssm.ops.triton.layer_norm import RMSNorm,layer_norm_fn,rms_norm_fn
 from torch.cuda.amp import custom_fwd,custom_bwd
 from causal_conv1d import causal_conv1d_fn
 from causal_conv1d.cpp_functions import causal_conv1d_fwd_function,causal_conv1d_bwd_function
 import causal_conv1d_cuda,selective_scan_cuda
 class ConvABI:
  @staticmethod
  def causal_conv1d_fwd(x,w,b,seq_idx,activation):
   return causal_conv1d_fwd_function(x,w,b,seq_idx,None,None,activation)
  @staticmethod
  def causal_conv1d_bwd(x,w,b,dy,seq_idx,dx,activation):
   return causal_conv1d_bwd_function(x,w,b,dy,seq_idx,None,None,dx,False,activation)[:3]
 env=dict(torch=torch,nn=nn,F=F,Tensor=Tensor,Optional=typing.Optional,List=typing.List,math=math,np=np,
          copy=copy,partial=partial,rearrange=rearrange,repeat=repeat,Mamba=Mamba,
          selective_scan_fn=selective_scan_fn,mamba_inner_fn=mamba_inner_fn,
          RMSNorm=RMSNorm,layer_norm_fn=layer_norm_fn,rms_norm_fn=rms_norm_fn,
          custom_fwd=custom_fwd,custom_bwd=custom_bwd,causal_conv1d_fn=causal_conv1d_fn,
          causal_conv1d_cuda=ConvABI,selective_scan_cuda=selective_scan_cuda)
 # DropPath is never entered with the published create_dfm_block default drop_path=0.
 env['DropPath']=lambda p: (_ for _ in ()).throw(ValueError('Nonzero DropPath requires actual timm implementation'))
 base=Path(repo)/'mld/models'
 for f in ['operator/bi_mamba_ops.py','operator/cross_attention.py','operator/mamba.py','operator/position_encoding.py','operator/position_encoding_layer.py','architectures/tools/embeddings.py']:
  definitions(base/f,env)
 env['lengths_to_mask']=lambda lengths,device:torch.arange(max(lengths),device=device)[None]<torch.as_tensor(lengths,device=device)[:,None]
 definitions(base/'architectures/mld_denoiser.py',env)
 return env

class ExternalDenoiser(nn.Module):
 def __init__(self,method,repo,motion_rep,hybrid):
  super().__init__();assert method=='motionmamba';self.method=method
  object.__setattr__(self,'motion_rep',motion_rep);object.__setattr__(self,'hybrid',hybrid)
  self.nframe_root_dim=20;self.latent_embedding_dim=128
  self.training_precision_contract='g1_root_output_geometry_fp32_v1';self.capacity=256
  env=official_environment(repo)
  self.input_proj=nn.Linear(3465,256)
  self.core=env['MldDenoiser'](ablation=SimpleNamespace(SKIP_CONNECT=True,VAE_TYPE='mld',DIFF_PE_TYPE='mld'),
       nfeats=148,condition='text',latent_dim=[256,256],ff_size=1024,num_layers=11,num_heads=4,
       dropout=.1,normalize_before=False,activation='gelu',position_embedding='learned',arch='dfm_enc',
       text_encoded_dim=768,guidance_scale=2.,guidance_uncondp=0.)
  # Widen only the text frontend; preserve the official768-D diffusion-time basis.
  self.core.text_encoded_dim=4096
  self.core.emb_proj=nn.Sequential(nn.ReLU(),nn.Linear(4096,256))
  blocks=list(self.core.encoder.input_blocks)+list(self.core.encoder.output_blocks)
  self.scan_counts=[b.mixer.number_of_scan for b in blocks]
  for b in blocks:
   b.mixer=BatchFirstScan(b.mixer);b.latent_mixer=BatchFirstScan(b.latent_mixer)
  self.output_proj=FP32Linear(256,148)
  # Same zero final prediction initialization used by the preceding common-G1 baselines.
  nn.init.zeros_(self.output_proj.weight);nn.init.zeros_(self.output_proj.bias)
  inactive=[]
  for name,p in self.named_parameters():
   unused=(name.startswith('core.mem_pos.') or '.norm2.' in name and '.middle_block.' not in name or '.norm3.' in name)
   if '.mixer.operation.' in name:
    suffix=name.split('.mixer.operation.',1)[1]
    if suffix.split('.')[0] in ['A_b_log','conv1d_b','x_proj_b','dt_proj_b','D_b']:unused=True
   if unused:inactive.append(name)
  self.inactive_prefixes=inactive;self.apply_trainable_policy()
  self.source=str(Path(repo)/'mld/models/operator/mamba.py');self.source_sha256=hashlib.sha256(Path(self.source).read_bytes()).hexdigest()
  self.source_edits=[
    'official11-layer symmetric encoder/decoder,256 hidden,HTM independent scan ensembles and bidirectional spatial Mamba retained; source commit c86ae037f6f9d07e0930143b6b29e374ff86c21b',
    'commonG1 hybrid148+414 observation/mask/role/heading pack replaces original learned2-token human VAE; official configurable latent count expanded to256 to reserve up to40s input; no token cropping or time compression',
    'right-zero-padding to256 input tokens followed by original time/text tokens; original learned positional embedding500 capacity retained; all source motion is accepted history/current noise, never hidden futureGT',
    'cached4096 text widens only original text projection, retaining official768-D time basis; common x0/four losses/noise/dropout/LR/batches replace official training frontend and original sampler',
    'DFMBlock sequence-first axes converted to batch-first at both Mamba calls; per-sample independence must pass before training',
    'legacy causal-conv1d ABI arguments bridged without changing convolution/scan arithmetic; isolated CUDA dependencies built from stock sources, global Native environment unchanged',
    'published unused norm2/norm3/mem_pos and unused base backward branch parameters remain saved but frozen to avoid DDP unused-gradient errors; active HTM/BSM parameters trained',
    'architecture adaptation, not a reproduction of original human-VAE MotionMamba system; no Native weights or persistent-state modules loaded']
 def apply_trainable_policy(self):
  for name,p in self.named_parameters():p.requires_grad_(name not in self.inactive_prefixes)
 def prepared(self,x,history_len,generation_len,future_len,history_mask,generation_mask,future_mask,
              history_token_mask,generation_token_mask,future_token_mask,text_feat,text_pad_mask,t,
              first_heading_angle,motion_mask=None,observed_motion=None):
  b,n,d=x.shape;assert d==148 and n<=self.capacity and bool((generation_len==52).all())
  assert observed_motion.shape==motion_mask.shape==(b,n*4,414)
  mask=motion_mask.float();obs=observed_motion.float()*mask
  role=torch.stack([history_token_mask,generation_token_mask,future_token_mask],-1).float();valid=role.bool().any(-1)
  angle=torch.stack([first_heading_angle.sin(),first_heading_angle.cos()],-1)
  a=torch.cat([x.float(),torch.cat([obs,mask],-1).reshape(b,n,3312),role,angle[:,None].expand(-1,n,-1)],-1)
  text=text_feat.float()[:,0]*text_pad_mask[:,0,None]
  return a,text,valid
 def forward(self,*args,**kwargs):
  keys=['x','history_len','generation_len','future_len','history_mask','generation_mask','future_mask','history_token_mask','generation_token_mask','future_token_mask','text_feat','text_pad_mask','t','first_heading_angle','motion_mask','observed_motion']
  bound=dict(zip(keys,args));bound.update(kwargs)
  if 'timesteps' in bound:bound['t']=bound.pop('timesteps')
  if 'text_feat_pad_mask' in bound:bound['text_pad_mask']=bound.pop('text_feat_pad_mask')
  a,text,valid=self.prepared(**bound);b,n,_=a.shape
  z=self.input_proj(a)*valid[:,:,None]
  z=F.pad(z,(0,0,0,self.capacity-n))
  z=self.core(z,bound['t'],text[:,None])[0]
  return self.output_proj(z[:,:n])
