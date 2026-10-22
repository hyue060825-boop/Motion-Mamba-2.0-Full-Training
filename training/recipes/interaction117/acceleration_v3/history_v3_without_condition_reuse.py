"""Training-only canonical history fast path; no persistent numerical cache.

All history tokens are accepted and there is no generation/future suffix. Keep
B1 projection shapes and all root arithmetic. Only structural layouts and
immutable host labels are cached. Current weights rebuild all recurrent states.
"""
import ast,inspect,types,collections
import torch
from einops import rearrange
from motion_training import history
from motion_training.acceleration import configure
from motion_mamba2.adapters.history_commit import build_canonical_history_layout
import speed_backend as speed

VERSION='canonical_history_only_v3'
_mode='dense';_layouts={};_host=collections.OrderedDict();_stats=collections.Counter()

source=inspect.getsource(speed.dense_commit_forward);tree=ast.parse(source)
for node in ast.walk(tree):
 if isinstance(node,ast.If) and 'not bool(x_pad_mask.all())' in ast.unparse(node.test):
  node.test=ast.parse('self.content_memory_read_view_runtime is not None or not getattr(self,"_v3_canonical_known_full",False)',mode='eval').body
ns=dict(speed.__dict__)
exec(compile(ast.fix_missing_locations(tree),'<trusted_canonical_dense_commit>','exec'),ns)
TRUSTED_DENSE=ns['dense_commit_forward']

def host_label(path):
 key=str(path)
 if key in _host:_host.move_to_end(key);_stats['host_cache_hit']+=1;return _host[key]
 item=torch.load(path,map_location='cpu',weights_only=False)
 leaves=[];groups=collections.defaultdict(list)
 def extract(x):
  if torch.is_tensor(x):
   i=len(leaves);leaves.append(x);groups[x.dtype].append(i);return ('__tensor_leaf__',i)
  if isinstance(x,dict):return {k:extract(v) for k,v in x.items()}
  if isinstance(x,list):return [extract(v) for v in x]
  if isinstance(x,tuple):return tuple(extract(v) for v in x)
  return x
 structure=extract(item);packed={};spec={}
 for dtype,indices in groups.items():
  packed[dtype]=torch.cat([leaves[i].detach().reshape(-1) for i in indices]).pin_memory()
  offset=0
  for i in indices:spec[i]=(dtype,offset,leaves[i].numel(),leaves[i].shape);offset+=leaves[i].numel()
 value=(structure,packed,spec);_host[key]=value
 if len(_host)>64:_host.popitem(last=False)
 _stats['host_cache_miss']+=1
 return value

def packed_item(path,device):
 structure,host,spec=host_label(path);buffers={dtype:x.to(device,non_blocking=True) for dtype,x in host.items()}
 views={i:buffers[dtype][off:off+n].view(shape) for i,(dtype,off,n,shape) in spec.items()}
 def restore(x):
  if isinstance(x,tuple) and len(x)==2 and isinstance(x[0],str) and x[0]=='__tensor_leaf__':return views[x[1]]
  if isinstance(x,dict):return {k:restore(v) for k,v in x.items()}
  if isinstance(x,list):return [restore(v) for v in x]
  if isinstance(x,tuple):return tuple(restore(v) for v in x)
  return x
 return restore(structure)

def layout(raw,tokens):
 key=(id(raw),tokens.device,tokens.dtype,tuple(tokens.shape))
 if key not in _layouts:
  _layouts[key]=build_canonical_history_layout(raw,tokens,history_frames=tokens.shape[1]*raw.num_frames_per_token,next_generation_frames=0)
 return _layouts[key]

def raw_history_forward(raw,tokens,l,kw):
 assert raw.motion_mask_mode=='concat' and not raw.sparsify_token_seq
 assert not raw.training and not torch.is_grad_enabled()
 b,t,_=tokens.shape;nfpt=raw.num_frames_per_token
 global_root,latent_body=raw.hybrid.get_root_and_latent_body_motion_from_hybrid(tokens)
 observed_root=raw.motion_rep.extract_root(l.observed_motion);root_mask=raw.motion_rep.extract_root(l.motion_mask)
 # Preserve the original arithmetic, including multiplication by zero, rather
 # than caching a weight-dependent root output or replacing it with GT root.
 global_root=global_root*(1-root_mask)+observed_root*root_mask
 token_index=torch.arange(t,device=tokens.device)[None,:].repeat(b,1)
 if raw.positional_encoding_mode=='learned_prefix_zero_at_first_generation':token_index=token_index-(l.history_len//nfpt)[:,None]
 else:assert raw.positional_encoding_mode=='default'
 # Original all-true fancy indexing produces these exact B1 flattened shapes.
 root_input=raw.global_root_hybrid_proj(tokens if raw.trt_compatible else tokens.reshape(b*t,-1)).reshape(b,t,raw.latent_dim)
 if raw.trt_compatible:
  # Original dense mask path adds two masked, finite zero projections.
  root_input=torch.zeros_like(root_input)+root_input*l.history_token_mask[:,:,None]+torch.zeros_like(root_input)
 assert root_input.dtype==raw.global_root_hybrid_constraints_proj.weight.dtype
 root_pred=raw.root_model(root_input,l.history_token_mask,kw['text_feat'],kw['text_pad_mask'],l.generation_len,kw['first_heading_angle'],token_index)
 root_pred=rearrange(root_pred,'b t (f d) -> b (t f) d',f=nfpt)
 root_pred=root_pred*l.generation_mask[:,:,None]+global_root*l.history_mask[:,:,None]
 local=raw.motion_rep.global_root_to_local_root(root_pred,normalized=True,lengths=l.history_len+l.generation_len)
 local=rearrange(local,'b (t f) d -> b t (f d)',f=nfpt)
 x_new=torch.cat([local,latent_body],-1)
 body_input=raw.local_root_hybrid_proj(x_new if raw.trt_compatible else x_new.reshape(b*t,-1)).reshape(b,t,raw.latent_dim)
 if raw.trt_compatible:body_input=torch.zeros_like(body_input)+body_input*l.history_token_mask[:,:,None]+torch.zeros_like(body_input)
 assert body_input.dtype==raw.local_root_hybrid_constraints_proj.weight.dtype
 raw.body_model(body_input,l.history_token_mask,kw['text_feat'],kw['text_pad_mask'],l.generation_len,kw['first_heading_angle'],token_index)

def prime(self,m,c,text):
 if _mode=='dense':
  for block in [c.controller.root_block,c.controller.body_block]:
   if block in speed._patched:block.forward=types.MethodType(speed.dense_commit_forward,block)
  return speed.speed_prime(self,m,c,text)
 raw=m.denoiser.model;was=raw.training;blocks=[c.controller.root_block,c.controller.body_block]
 raw.eval();c.reset_memory();configure(c,'optimized')
 for block in blocks:
  block.sequence_stack.inference_graph=False;block._speed_dense_prime=True;block._v3_canonical_known_full=True
  if block not in speed._patched:speed._patched[block]=block.forward
  block.forward=types.MethodType(TRUSTED_DENSE,block)
 try:
  with torch.no_grad(),torch.autocast(device_type=m.device.type if isinstance(m.device,torch.device) else 'cuda',enabled=False):
   for tokens,kw in self.chunks:
    assert kw['include_synthetic_generation'] is False and kw.get('detach',True)
    l=layout(raw,tokens)
    for block in blocks:block.prepare_accepted_commit(l.history_token_mask,chunk_id=kw['chunk_id'],detach=True)
    before=[block.accepted_commits for block in blocks]
    raw_history_forward(raw,tokens,l,kw)
    assert all(block.accepted_commits==n+1 for block,n in zip(blocks,before))
    for block in blocks:block.cancel_accepted_commit()
  ctl=c.controller;n=len(self.chunks)
  ctl._completed_windows=ctl._accepted_commits=ctl._next_chunk_id=n
  ctl._last_text_feat=text;ctl._last_text_pad_mask=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool)
  return dict(history_frames=int(self.physical.shape[1]),prime_chunks=n,physical_hash=speed.diagnostic_identity(self.physical),collector=self.collector_checkpoint_sha256,weight_state_rebuilt=True,cached_numerical_state_reused=False,explicit_history_frames=0,duplicated_frames=0)
 except Exception:
  # This is a whole fresh training prefix, not an online partially committed
  # session. Abandon the prefix and raise before any optimizer update.
  c.reset_memory();raise
 finally:
  raw.train(was)
  for block in blocks:block._speed_dense_prime=False;block._v3_canonical_known_full=False;block.cancel_accepted_commit()
  configure(c,'reference')

def install(mode='canonical',audit=False):
 global _mode
 assert mode in ['dense','canonical'];_mode=mode
 speed.install('dense',audit=audit)
 speed.PACKED_BATCH.__globals__['packed_item']=speed.packed_item if mode=='dense' else packed_item
 history.HistoryContext.prime=prime

def stats():return dict(_stats,layout_templates=len(_layouts),host_labels=len(_host))
