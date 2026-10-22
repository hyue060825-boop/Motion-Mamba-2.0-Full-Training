"""History-only layer pre/post graphs. Mixer/cache commits remain outside.

Extract numerical expressions from the actual trusted layer forward. Live
parameter storage is captured, not weights or recurrent state from a prior
update. No current-window training forward/backward uses these graphs.
"""
import ast, collections, inspect, textwrap, types
import torch
from motion_mamba2.kernels.dit.layer import DiTSequenceLayer
import extra_graph

_enabled=False; _original={}; _entries={}; _stats=collections.Counter()

source=ast.parse(textwrap.dedent(inspect.getsource(DiTSequenceLayer.forward)))
node=source.body[0]
split=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
           and ast.unparse(n.targets[0])=='(mixed, pending, aux)')
ns=dict(DiTSequenceLayer.forward.__globals__)
from motion_mamba2.kernels.dit.fused_pointwise import modulated_norm, residual
ns.update(modulated_norm=modulated_norm,residual=residual)
def compile_part(name, names, body):
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in names],
        kwonlyargs=[],kw_defaults=[],defaults=[])
    fn=ast.FunctionDef(name=name,args=args,body=body,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
         '<unchanged_layer_numeric_'+name+'>','exec'),ns)
    return ns[name]

pre=compile_part('_pre',['self','x','condition'],node.body[:split]+[
    ast.parse('return mixer_input, gate_mixer, shift_mlp, scale_mlp, gate_mlp').body[0]])
post=compile_part('_post',['self','x','mixed','gate_mixer','shift_mlp','scale_mlp','gate_mlp','current_mask'],
    node.body[split+1:-1]+[ast.parse('return x').body[0]])

def install(c):
    for block in [c.controller.root_block,c.controller.body_block]:
        for layer in block.sequence_stack.layers:
            if layer not in _original:
                _original[layer]=layer.forward
                layer.forward=types.MethodType(forward,layer)

def enable(value):
    global _enabled
    _enabled=value;_stats.clear()

def graph_call(layer,phase,inputs):
    params=tuple((id(p),p.data_ptr(),tuple(p.shape),p.dtype,p.device) for p in layer.parameters())
    sig=tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in inputs)
    flags=(getattr(layer,'inference_fused_adaln',False),getattr(layer,'inference_fused_pointwise',False))
    key=(id(layer),phase,params,sig,flags)
    entry=_entries.get(key)
    fn=pre if phase=='pre' else lambda self,*a:post(self,*a,None)
    if entry is None:
        if sum(k[0]==id(layer) and k[1]==phase for k in _entries)>=4:
            _stats['shape_fallback']+=1;return fn(layer,*inputs)
        static=[torch.empty_strided(t.size(),t.stride(),dtype=t.dtype,device=t.device).copy_(t) for t in inputs]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        # Use original MLP/condition kernels inside this larger numerical graph,
        # rather than replay nested CUDA graphs from extra_graph.
        previous=extra_graph._enabled;extra_graph._enabled=False
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):fn(layer,*static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn(layer,*static)
        finally:extra_graph._enabled=previous
        entry=dict(inputs=static,out=out,graph=graph);_entries[key]=entry;_stats['captures']+=1
    else:
        for dst,src in zip(entry['inputs'],inputs):dst.copy_(src)
    entry['graph'].replay();_stats['replays']+=1
    out=entry['out']
    return tuple(t.clone() for t in out) if isinstance(out,tuple) else out.clone()

def forward(self,x,condition,*,state,do_read=True,commit=False,chunk_id=None,
    current_mask=None,attention_mask=None,detach_on_commit=True,
    content_memory_read_enabled=None,content_memory_write_enabled=True,
    content_memory_delta_mode='normal',content_memory_condition=None,
    content_memory_read_view=None,content_memory_read_view_level='slots',
    compute_pending=True,collect_diagnostics=True):
    kw=dict(state=state,do_read=do_read,commit=commit,chunk_id=chunk_id,
        current_mask=current_mask,attention_mask=attention_mask,
        detach_on_commit=detach_on_commit,content_memory_read_enabled=content_memory_read_enabled,
        content_memory_write_enabled=content_memory_write_enabled,content_memory_delta_mode=content_memory_delta_mode,
        content_memory_condition=content_memory_condition,content_memory_read_view=content_memory_read_view,
        content_memory_read_view_level=content_memory_read_view_level,compute_pending=compute_pending,
        collect_diagnostics=collect_diagnostics)
    if (not _enabled or self.training or torch.is_grad_enabled() or not x.is_cuda
        or not commit or current_mask is not None):
        return _original[self](x,condition,**kw)
    mixer_input,gate_mixer,shift_mlp,scale_mlp,gate_mlp=graph_call(self,'pre',[x,condition])
    # The existing mixer still validates live projection versions and performs
    # every accepted commit, cache eviction and recurrent-state update.
    mixed,pending,aux=self.mixer.forward_chunk(mixer_input,**kw)
    out=graph_call(self,'post',[x,mixed,gate_mixer,shift_mlp,scale_mlp,gate_mlp])
    return out,pending,aux

def stats():return dict(_stats,entries=len(_entries))
