"""Numerical front-end graph used only in detached, eval-mode history."""
import types,collections,inspect,ast,textwrap
import torch
_enabled=False;_original={};_core={};_parameters={};_entries={};_stats=collections.Counter()
def install(c):
 for block in [c.controller.root_block,c.controller.body_block]:
  if block not in _original:
   _original[block]=block._encode
   # Split original source at positional encoding. Keep its validity assertion
   # and optional dropout outside capture on every invocation, unchanged.
   original=block._encode
   node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
   cuts=[i for i,n in enumerate(node.body) if isinstance(n,ast.If) and 'positional_encoding_mode' in ast.unparse(n.test)]
   assert len(cuts)==1
   node.body=node.body[:cuts[0]]+[ast.parse('return motion,condition,memory_condition').body[0]]
   scope=dict(original.__func__.__globals__)
   exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original_encode_numeric_core>','exec'),scope)
   _core[block]=types.MethodType(scope[node.name],block)
   _parameters[block]=[p for name,p in block.named_parameters() if not name.startswith('sequence_stack.')]
   block._encode=types.MethodType(encode,block)
def enable(value):
 global _enabled
 _enabled=value;_stats.clear()
def encode(self,*args):
 original=_original[self]
 if not _enabled or self.training or torch.is_grad_enabled() or not args[0].is_cuda:return original(*args)
 assert all(torch.is_tensor(x) or x is None for x in args)
 shape=tuple((tuple(x.shape),tuple(x.stride()),x.dtype,x.device) if x is not None else None for x in args)
 params=tuple((id(p),p.data_ptr(),p.dtype,p.device) for p in _parameters[self])
 flags=(self.num_text_tokens,self.use_text_mask,self.input_first_heading_angle,self.positional_encoding_mode)
 key=(id(self),shape,params,flags)
 entry=_entries.get(key)
 if entry is None:
  if sum(k[0]==id(self) for k in _entries)>=4:return original(*args)
  static=tuple(x.clone() if x is not None else None for x in args)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(2):_core[self](*static)
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out=_core[self](*static)
  entry=dict(args=static,out=out,graph=graph);_entries[key]=entry;_stats['captures']+=1
 else:
  for x,y in zip(entry['args'],args):
   if x is not None:x.copy_(y)
 entry['graph'].replay();_stats['replays']+=1
 motion,condition,memory_condition=(x.clone() for x in entry['out'])
 if self.positional_encoding_mode=='default':motion=self.sequence_pos_encoder(motion)
 else:
  if args[-1] is None:raise ValueError('token_index is required for negative-index encoding')
  motion=self.motion_token_embedding(motion,args[-1])
 valid=args[1].to(device=motion.device,dtype=torch.bool)
 motion=motion*valid[:,:,None].to(motion.dtype)
 return motion,condition,valid,memory_condition
def stats():return dict(_stats,entries=len(_entries))
