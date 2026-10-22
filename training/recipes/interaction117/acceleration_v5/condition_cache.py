"""Exact prompt/clean-time conditioning reuse WITHIN one optimizer update.

Motion projections always recompute. Reuse is limited to eval+no-grad accepted
history. Full current weights must retain their version until end_update.
"""
import collections,types
import torch
from motion_training.io import identity
import history_v3
import speed_backend
_enabled=False;_encode_original={};_mod_original={};_cpu_keys={};_conditions={};_modulations={};_stats=collections.Counter();_pending={};_raw=None;_versions=None
_orig_history_forward=history_v3.raw_history_forward

def cpu_keys(path):
 if path not in _cpu_keys:
  item=speed_backend._cpu_labels.get(path)
  if item is None:item=torch.load(path,map_location='cpu',weights_only=False)
  _cpu_keys[path]=[]
  for tokens,kw in item['chunks']:
   base={k:kw[k] for k in ['text_feat','text_pad_mask']}
   _cpu_keys[path].append((identity(base),identity(dict(base,heading=kw.get('first_heading_angle')))))
 return _cpu_keys[path]

def versions(raw):return tuple((id(p),p.data_ptr(),p._version) for p in raw.parameters())

def begin(records,enabled,raw=None):
 global _enabled,_pending,_raw,_versions
 _raw=raw;_versions=versions(raw) if enabled else None
 if enabled:assert raw is not None
 _enabled=enabled;_conditions.clear();_modulations.clear();_stats.clear();_pending={}
 if enabled:
  for r in records:
   path=str(r['label']);keys=cpu_keys(path)
   # GPU tokens are packed immutable copies. Route each chunk by its scalar
   # index at prime time, using the packed label's precomputed CPU keys below.
   _pending[path]=keys

_original_packed=speed_backend.packed_item
_label=None

def packed(path,device):
 global _label
 _label=str(path)
 return _original_packed(path,device)

_cursor=0
_original_context_init=None

def install(c):
 global _original_context_init
 from motion_training import history
 if _original_context_init is None:
  _original_context_init=history.HistoryContext.__init__
  def init(self,*args,**kwargs):
   global _cursor
   if _enabled:assert versions(_raw)==_versions,"Weights changed inside the conditioning reuse update"
   _original_context_init(self,*args,**kwargs);_cursor=0
  history.HistoryContext.__init__=init
 speed_backend.packed_item=packed
 speed_backend.PACKED_BATCH.__globals__['packed_item']=packed
 history_v3.raw_history_forward=raw_history_forward
 for block in [c.controller.root_block,c.controller.body_block]:
  if block not in _encode_original:
   _encode_original[block]=block._encode;block._encode=types.MethodType(encode,block)
  for layer in block.sequence_stack.layers:
   module=layer.condition_modulation
   if module not in _mod_original:
    _mod_original[module]=module.forward;module.forward=types.MethodType(modulation,module)

_active_keys=None

def raw_history_forward(raw,tokens,layout,kw):
 global _active_keys,_cursor
 _active_keys=_pending[_label][_cursor] if _enabled else None
 _cursor+=1
 try:return _orig_history_forward(raw,tokens,layout,kw)
 finally:_active_keys=None


def encode(self,x,x_pad,text,text_mask,timesteps,heading,token_index):
 original=_encode_original[self]
 if (not _enabled or self.training or torch.is_grad_enabled() or _active_keys is None):
  return original(x,x_pad,text,text_mask,timesteps,heading,token_index)
 # All admitted canonical history uses clean timestep0, B1, and fixed flags.
 assert timesteps.shape==(1,)
 key=(id(self),_active_keys[1 if self.input_first_heading_angle else 0],self.num_text_tokens,self.use_text_mask,self.input_first_heading_angle,self.positional_encoding_mode,x.dtype)
 cached=_conditions.get(key)
 if cached is None:
  motion,condition,valid,memory=original(x,x_pad,text,text_mask,timesteps,heading,token_index)
  _conditions[key]=(condition,memory);_stats['condition_misses']+=1
  return motion,condition,valid,memory
 condition,memory=cached
 motion=self.input_linear(x)
 if self.positional_encoding_mode=='default':motion=self.sequence_pos_encoder(motion)
 else:motion=self.motion_token_embedding(motion,token_index)
 valid=x_pad.to(device=motion.device,dtype=torch.bool)
 motion=motion*valid[:,:,None].to(motion.dtype)
 _stats['condition_hits']+=1
 return motion,condition,valid,memory


def modulation(self,condition):
 original=_mod_original[self]
 if not _enabled or self.training or torch.is_grad_enabled():return original(condition)
 key=(id(self),condition.data_ptr(),condition._version,tuple(condition.shape),condition.dtype,condition.device)
 cached=_modulations.get(key)
 if cached is not None:
  assert cached[0] is condition
  _stats['modulation_hits']+=1;return cached[1]
 out=original(condition)
 # Retain the input so its storage address cannot be reused within the update.
 _modulations[key]=(condition,out);_stats['modulation_misses']+=1
 return out

def end():
 global _enabled,_raw,_versions
 _conditions.clear();_modulations.clear();_enabled=False;_raw=None;_versions=None
def stats():return dict(_stats,condition_entries=len(_conditions),modulation_entries=len(_modulations))
