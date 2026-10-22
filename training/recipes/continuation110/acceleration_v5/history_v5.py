"""Isolated no-grad history candidates; current gradient path is unchanged.

Prefixes may only be reused inside one update with fixed parameter versions.
Mamba graphs use live parameter storage and refresh every input/state on replay.
Neither optimization runs in gradient-enabled current-window computation.
"""
from common import *
import collections,copy,types
import torch
from motion_training import history
from motion_training.acceleration import configure
import history_v3 as v3
import speed_backend as speed

_mode='current';_counts={};_wanted=set();_label_keys={};_prefix_cache={};_weights=None;_stats=collections.Counter();_graphs={};_wrappers={}
_FIELDS=('angle_dt_state','ssm_state','k_state','v_state')

def tensor_size(obj):
 if torch.is_tensor(obj):return obj.numel()*obj.element_size()
 if isinstance(obj,dict):return sum(tensor_size(x) for x in obj.values())
 if isinstance(obj,(tuple,list)):return sum(tensor_size(x) for x in obj)
 if hasattr(obj,'__dict__'):return tensor_size(vars(obj))
 return 0

def versions(raw):return tuple((id(p),p._version,p.data_ptr()) for p in raw.parameters())

def chunk_keys(path):
 if path not in _label_keys:
  if path in speed._cpu_labels:item=speed._cpu_labels[path]
  else:item=torch.load(path,map_location='cpu',weights_only=False)
  _label_keys[path]=tuple(identity(chunk) for chunk in item['chunks'])
 return _label_keys[path]

def begin_update(m,c,records,mode):
 global _mode,_counts,_weights,_wanted
 _mode=mode;_prefix_cache.clear();_counts=collections.Counter();_weights=versions(m.denoiser.model)
 if mode in ('prefix','prefix_graph'):
  for r in records:
   keys=chunk_keys(str(r['label']))
   for n in range(1,len(keys)+1):_counts[keys[:n]]+=1
 children=collections.defaultdict(int)
 for key,count in _counts.items():children[key[:-1]]=max(children[key[:-1]],count)
 _wanted={key for key,count in _counts.items() if count>1 and children[key]<count}
 _stats.clear()
 for block in [c.controller.root_block,c.controller.body_block]:
  for layer in block.sequence_stack.layers:
   wrapper=layer.mixer._mamba_wrapper
   if wrapper is not None and wrapper not in _wrappers:
    _wrappers[wrapper]=wrapper.forward_chunk
    wrapper.forward_chunk=types.MethodType(graph_forward,wrapper)
 if mode in ('prefix','prefix_graph'):history.HistoryContext.prime=prime


def graph_forward(self,x,state=None,reset_mask=None):
 original=_wrappers[self]
 if (_mode not in ('graph','prefix_graph') or self.mamba3.training or torch.is_grad_enabled()
     or not x.is_cuda or state is None or reset_mask is not None):
  return original(x,state,reset_mask)
 inputs=[x]+[getattr(state,k) for k in _FIELDS]
 params=tuple((id(p),p.data_ptr(),tuple(p.shape),p.dtype,p.device) for p in self.mamba3.parameters())
 signature=tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in inputs)
 key=(id(self),params,signature,bool(getattr(self,'inference_fused_adt',False)))
 entry=_graphs.get(key)
 if entry is None:
  # This graph is numerical only: no cache commit, RNG, or Python state change.
  if sum(k[0]==id(self) for k in _graphs)>=4:return original(x,state,reset_mask)
  sx=x.clone();ss=state.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(2):original(sx,ss,None)
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out,next_state=original(sx,ss,None)
  entry=dict(x=sx,state=ss,out=out,next_state=next_state,graph=graph)
  _graphs[key]=entry;_stats['graph_captures']+=1
 else:
  entry['x'].copy_(x)
  for k in _FIELDS:getattr(entry['state'],k).copy_(getattr(state,k))
 entry['graph'].replay();_stats['graph_replays']+=1
 # Every returned state owns its storage; the next replay cannot overwrite it.
 return entry['out'].clone(),entry['next_state'].clone()


def prime(self,m,c,text):
 raw=m.denoiser.model
 assert versions(raw)==_weights,'Parameters changed within the update prefix cache'
 keys=tuple(identity(chunk) for chunk in self.chunks) if not hasattr(self,'_v5_prefix_keys') else self._v5_prefix_keys
 # Keys normally supplied from immutable CPU labels; GPU hashing is a diagnostic
 # fallback only and is deliberately not used in production.
 was=raw.training;blocks=[c.controller.root_block,c.controller.body_block]
 raw.eval();c.reset_memory();configure(c,'optimized')
 for block in blocks:
  block.sequence_stack.inference_graph=False;block._speed_dense_prime=True;block._v3_canonical_known_full=True
  if block not in speed._patched:speed._patched[block]=block.forward
  block.forward=types.MethodType(v3.TRUSTED_DENSE,block)
 hit=0
 try:
  with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
   for n in range(len(keys),0,-1):
    if keys[:n] in _prefix_cache:
     saved=_prefix_cache[keys[:n]]
     for block,state in zip(blocks,saved):block._sequence_state=copy.deepcopy(state)
     hit=n;_stats['reused_chunks']+=n;break
   cc=sys.modules.get('condition_cache')
   if cc is not None:cc.seek_history_prefix(hit)
   for n,(tokens,kw) in enumerate(self.chunks[hit:],start=hit+1):
    assert kw['include_synthetic_generation'] is False and kw.get('detach',True)
    layout=v3.layout(raw,tokens)
    for block in blocks:block.prepare_accepted_commit(layout.history_token_mask,chunk_id=kw['chunk_id'],detach=True)
    before=[block.accepted_commits for block in blocks]
    v3.raw_history_forward(raw,tokens,layout,kw)
    assert all(block.accepted_commits==count+1 for block,count in zip(blocks,before))
    for block in blocks:block.cancel_accepted_commit()
    _stats['recomputed_chunks']+=1
    key=keys[:n]
    if key in _wanted and key not in _prefix_cache:
     states=[block._sequence_state for block in blocks]
     needed=tensor_size(states)
     if sum(tensor_size(s) for s in _prefix_cache.values())+needed<=2*1024**3:
      _prefix_cache[key]=copy.deepcopy(states)
   ctl=c.controller;n=len(self.chunks)
   ctl._completed_windows=ctl._accepted_commits=ctl._next_chunk_id=n
   ctl._last_text_feat=text;ctl._last_text_pad_mask=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool)
   return dict(history_frames=int(self.physical.shape[1]),prime_chunks=n,physical_hash=speed.diagnostic_identity(self.physical),collector=self.collector_checkpoint_sha256,weight_state_rebuilt=True,cached_numerical_state_reused=False,within_update_reused_chunks=hit,explicit_history_frames=0,duplicated_frames=0)
 except Exception:c.reset_memory();raise
 finally:
  raw.train(was)
  for block in blocks:block._speed_dense_prime=False;block._v3_canonical_known_full=False;block.cancel_accepted_commit()
  configure(c,'reference')

# HistoryContext has no label path. Route precomputed exact CPU chunk identities
# by packed_item creation, before the context is constructed, without GPU hashes.
def install_prefix_keys():
 original=history.HistoryContext.__init__
 if getattr(original,'_v5_key_route',False):return
 def init(self,chunks,physical,collector):
  original(self,chunks,physical,collector)
  keys=_pending_keys
  if keys is not None and len(keys)==len(chunks):self._v5_prefix_keys=keys
 init._v5_key_route=True;history.HistoryContext.__init__=init

_pending_keys=None
_original_packed=speed.packed_item

def routed_packed(path,device):
 global _pending_keys
 _pending_keys=chunk_keys(str(path)) if _mode in ('prefix','prefix_graph') else None
 return _original_packed(path,device)


def install():
 install_prefix_keys()
 speed.packed_item=routed_packed
 # Legacy builder is the frozen AST function; replace its loader binding too.
 speed.PACKED_BATCH.__globals__['packed_item']=routed_packed

def end_update():
 _prefix_cache.clear()

def stats():return dict(_stats,graph_entries=len(_graphs),prefix_cache_bytes=tensor_size(_prefix_cache))
