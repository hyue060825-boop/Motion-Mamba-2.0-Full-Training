"""Isolated exact-history optimization; never used for current gradient forward."""
from common import *
import ast,functools,collections,types
import torch
from motion_training import history
from motion_training.acceleration import configure

ORIGINAL_BATCH=history.teacher_batch
ORIGINAL_PRIME=history.HistoryContext.prime
ORIGINAL_IDENTITY=history.identity
_audit=True
_mode='reference'
_patched={}
_cpu_labels=collections.OrderedDict()

def diagnostic_identity(value):
    return ORIGINAL_IDENTITY(value) if _audit else 'deferred_to_periodic_audit'

def packed_item(path,device):
    """Three host transfers instead of hundreds of small blocking transfers.

    Every tensor leaf gets its own disjoint view, as in recursive copy_state.
    Input data are immutable; copies are fresh on each call, not stale states.
    """
    key=str(path)
    if key not in _cpu_labels:
        _cpu_labels[key]=torch.load(path,map_location='cpu',weights_only=False)
        if len(_cpu_labels)>64:_cpu_labels.popitem(last=False)
    item=_cpu_labels[key];_cpu_labels.move_to_end(key)
    leaves=[];by_dtype=collections.defaultdict(list)
    def extract(x):
        if torch.is_tensor(x):
            i=len(leaves);leaves.append(x);by_dtype[x.dtype].append(i)
            return ('__tensor_leaf__',i)
        if isinstance(x,dict):return {k:extract(v) for k,v in x.items()}
        if isinstance(x,list):return [extract(v) for v in x]
        if isinstance(x,tuple):return tuple(extract(v) for v in x)
        return x
    structure=extract(item);views={}
    for dtype,indices in by_dtype.items():
        host=torch.cat([leaves[i].detach().reshape(-1) for i in indices])
        packed=host.pin_memory().to(device,non_blocking=True);offset=0
        for i in indices:
            t=leaves[i];views[i]=packed[offset:offset+t.numel()].view(t.shape);offset+=t.numel()
    def restore(x):
        if isinstance(x,tuple) and len(x)==2 and x[0]=='__tensor_leaf__':return views[x[1]]
        if isinstance(x,dict):return {k:restore(v) for k,v in x.items()}
        if isinstance(x,list):return [restore(v) for v in x]
        if isinstance(x,tuple):return tuple(restore(v) for v in x)
        return x
    return restore(structure)

def dense_commit_forward(self,x,x_pad_mask,text_feat,text_feat_pad_mask,timesteps,first_heading_angle=None,token_index=None):
    original=_patched[self]
    if not getattr(self,'_speed_dense_prime',False) or self.training or torch.is_grad_enabled() or self._accepted_commit is None:
        return original(x,x_pad_mask,text_feat,text_feat_pad_mask,timesteps,first_heading_angle,token_index)
    # Only canonical history-only, all-valid, fully accepted chunks are admitted.
    selection,chunk_id,detach=self._accepted_commit
    if self.content_memory_read_view_runtime is not None or not bool(x_pad_mask.all()) or not torch.equal(selection,x_pad_mask):
        return original(x,x_pad_mask,text_feat,text_feat_pad_mask,timesteps,first_heading_angle,token_index)
    motion,condition,valid,memory_condition=self._encode(x,x_pad_mask,text_feat,text_feat_pad_mask,timesteps,first_heading_angle,token_index)
    parameter=self._stack_parameter();motion=motion.to(parameter);condition=condition.to(parameter);memory_condition=memory_condition.to(parameter);valid=valid.to(parameter.device)
    state=self._ensure_state(x.shape[0]) if self._sequence_state is None else self._sequence_state
    assert state.layers[0].attention_cache.batch_size==x.shape[0]
    # The original computes the same stack twice: read-only, then accepted commit.
    # Reuse the commit's identical hidden output. Commit order/state stays intact.
    hidden,_,_=self.sequence_stack(motion,condition,state=state,commit=True,chunk_id=chunk_id,current_mask=None,detach_on_commit=detach,collect_diagnostics=False,
        content_memory_read_enabled=self.content_memory_read_enabled_runtime,content_memory_write_enabled=self.content_memory_write_enabled_runtime,
        content_memory_delta_mode=self.content_memory_delta_mode_runtime,content_memory_condition=memory_condition,content_memory_read_view=None,content_memory_read_view_level='slots')
    for layer in self.sequence_stack.layers:
        layer.mixer.attention._inference_read_scope=None
        layer.mixer.attention._inference_history_cache=None
    self.read_only_forwards+=1;self.accepted_commits+=1;self._accepted_commit=None
    if getattr(self,'output_projection_fp32',False):
        with torch.autocast(device_type=hidden.device.type,enabled=False):out=self.output_linear(hidden.float())
    else:out=self.output_linear(hidden.to(self.output_linear.weight.dtype))
    return out*valid[:,:,None].to(out.dtype)

def speed_prime(self,m,c,text):
    blocks=[c.controller.root_block,c.controller.body_block]
    configure(c,'optimized')
    # Read-only graphs are version-keyed and would be rebuilt after every update.
    # Here only existing fused inference kernels and unnecessary diagnostics off.
    for b in blocks:
        b.sequence_stack.inference_graph=False
        b._speed_dense_prime=_mode=='dense'
        if b not in _patched:
            _patched[b]=b.forward;b.forward=types.MethodType(dense_commit_forward,b)
    try:return ORIGINAL_PRIME(self,m,c,text)
    finally:
        for b in blocks:b._speed_dense_prime=False
        configure(c,'reference')

# Keep the exact original target/noise/mask/dropout/decode implementation.
tree=ast.parse((ROOT/'code/src/motion_training/history.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='teacher_batch')
class Loader(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='copy_state' and node.args and isinstance(node.args[0],ast.Call) and isinstance(node.args[0].func,ast.Attribute) and node.args[0].func.attr=='load':
            return ast.copy_location(ast.Call(func=ast.Name(id='packed_item',ctx=ast.Load()),args=[node.args[0].args[0],node.args[1]],keywords=[]),node)
        return node
fn=Loader().visit(fn);ns=dict(history.__dict__,packed_item=packed_item,identity=diagnostic_identity)
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<packed_original_teacher_batch>','exec'),ns)
PACKED_BATCH=ns['teacher_batch']

def install(mode='reference',audit=False):
    global _mode,_audit
    assert mode in ('reference','kernels','dense');_mode=mode;_audit=audit
    history.identity=ORIGINAL_IDENTITY if mode=='reference' else diagnostic_identity
    history.HistoryContext.prime=ORIGINAL_PRIME if mode=='reference' else speed_prime
    history.teacher_batch=ORIGINAL_BATCH if mode=='reference' else PACKED_BATCH

def set_audit(value):
    global _audit
    _audit=value
