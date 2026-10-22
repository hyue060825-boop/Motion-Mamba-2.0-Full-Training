from common import *
import copy,random,types
import numpy as np
import torch
from motion_training import runtime
from motion_training._compat import legacy_loss
from g1_motion_contract import inspect_stats,validate_checkpoint
from g1_padding_contract import install_padding_contract
from g1_root_body_gradient import configure_root_body_gradient
from g1_training_precision import configure_training_precision,FP32_BOUNDARIES
from motion_mamba2.models.backbone import TransformerEncoderBlock
from motion_training.io import identity

class Client:
 def reset_memory(self):pass

def load(cp,training=False,device='cuda',fresh_seed=None,official_frontend=False):
 torch.set_num_threads(2)
 ck=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
 cfg=ck['config'];validate_checkpoint(ck,inspect_stats(cfg['stats_dir']))
 seed=123 if fresh_seed is None else fresh_seed
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 # Never load the released Transformer weights: construct the entire denoiser.
 m=runtime.load_model('g1',device=device,text_encoder=False,fresh_denoiser=True)
 raw=m.denoiser.model
 assert isinstance(raw.root_model,TransformerEncoderBlock) and isinstance(raw.body_model,TransformerEncoderBlock)
 assert not any('mamba' in n or 'sequence_stack' in n for n,_ in raw.named_parameters())
 if fresh_seed is not None:
  parent=torch.load(OLD/'initial/checkpoint0.pt',map_location='cpu',weights_only=False,mmap=True)
  keys=read(OLD/'INITIALIZATION_VERIFIED.json')['official_compatible_keys']
  source=dict(parent['denoiser_aux'])
  source.update({'root_model.'+n:v for n,v in parent['root_model'].items()})
  source.update({'body_model.'+n:v for n,v in parent['body_model'].items()})
  state=raw.state_dict()
  for n in keys:
   assert n in state and state[n].shape==source[n].shape,n
   state[n].copy_(source[n])
   assert torch.equal(state[n].float().cpu(),source[n].float()),n
  ck=copy.copy(ck);ck['config']=dict(cfg,mode='ardy',original_initialization='matched_fresh_transformer',content_memory_enabled=False)
  ck['step']=0;ck['_framework_global_start']=0
  ck['initialization']=dict(policy='fresh_denoiser',seed=seed,architecture='official_motion_mamba2_transformer',official_sequence_weights_loaded=False,shared_native_checkpoint=str(OLD/'initial/checkpoint0.pt'),shared_keys=keys,trainable_sequence_initialized_by_constructor=True)
 else:
  assert ck['initialization']['architecture']=='official_motion_mamba2_transformer'
  raw.root_model.load_state_dict(ck['root_model'],strict=True);raw.body_model.load_state_dict(ck['body_model'],strict=True)
  for n,v in ck['denoiser_aux'].items():raw.state_dict()[n].copy_(v)
 legacy_loss._replace_stats(m,cfg['stats_dir'])
 m.training_codec_contract=cfg.get('training_codec_contract','legacy');m.checkpoint_training_manifest=cfg.get('manifest')
 m.generation_contract=cfg.get('generation_contract','legacy')
 install_padding_contract(m,cfg['padding_contract']);configure_root_body_gradient(raw,cfg.get('root_body_gradient_scale',0))
 configure_training_precision(raw,m.autoencoder,FP32_BOUNDARIES)
 m.device=next(raw.parameters()).device;m.eval()
 for p in m.parameters():p.requires_grad_(False)
 for p in raw.parameters():p.requires_grad_(training)
 return m,Client(),ck

def install():runtime.load=load
