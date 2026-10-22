"""History-only numerical graphs for existing MLP/condition modules.

Live parameter storage, unchanged module math, gradient/training fallback.
No attention cache, accepted commit, RNG, or Python metadata inside capture.
"""
import types,collections
import torch
_enabled=False;_original={};_entries={};_stats=collections.Counter()

def install(c):
 for block in [c.controller.root_block,c.controller.body_block]:
  for layer in block.sequence_stack.layers:
   for module in [layer.mlp,layer.condition_modulation]:
    if module not in _original:
     _original[module]=module.forward;module.forward=types.MethodType(forward,module)

def enable(value):
 global _enabled
 _enabled=value;_stats.clear()

def forward(self,x):
 original=_original[self]
 if not _enabled or self.training or torch.is_grad_enabled() or not x.is_cuda:return original(x)
 params=tuple((id(p),p.data_ptr(),tuple(p.shape),p.dtype,p.device) for p in self.parameters())
 key=(id(self),params,tuple(x.shape),tuple(x.stride()),x.dtype,x.device)
 entry=_entries.get(key)
 if entry is None:
  if sum(k[0]==id(self) for k in _entries)>=4:return original(x)
  sx=x.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(2):original(sx)
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out=original(sx)
  entry=dict(x=sx,out=out,graph=graph);_entries[key]=entry;_stats['captures']+=1
 else:entry['x'].copy_(x)
 entry['graph'].replay();_stats['replays']+=1
 return entry['out'].clone()

def stats():return dict(_stats,entries=len(_entries))
