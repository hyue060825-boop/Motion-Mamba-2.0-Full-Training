"""Reuse literal metadata only. Dynamic masks, math and parameters are unchanged."""
import ast, copy, hashlib, types
from pathlib import Path
import torch
import torch.nn.functional as F


def cached_attention(original, enabled):
    causal_masks = {}
    def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., causal=False,
                  num_frame_per_block=1, dtype=torch.float):
        if not enabled[0]:
            return original(q, k, v, q_lens, k_lens, dropout_p, causal, num_frame_per_block, dtype)
        b, lq, lk = q.size(0), q.size(1), k.size(1)
        device=q.device;blocked=None
        # Real lengths are rebuilt every call. They are never cached as valid.
        if not(q_lens is None and k_lens is None):
            q_lens=(torch.full((b,),lq,dtype=torch.int32,device=device) if q_lens is None
                    else q_lens.to(device=device,dtype=torch.int32))
            k_lens=(torch.full((b,),lk,dtype=torch.int32,device=device) if k_lens is None
                    else k_lens.to(device=device,dtype=torch.int32))
            qp=torch.arange(lq,device=device,dtype=torch.int32)[None]>=q_lens[:,None]
            kp=torch.arange(lk,device=device,dtype=torch.int32)[None]>=k_lens[:,None]
            blocked=qp[:,None,:,None]|kp[:,None,None,:]
        if causal:
            assert lq==lk
            key=(lq,lk,num_frame_per_block,str(device))
            if key not in causal_masks:
                qi=torch.arange(lq,device=device)[:,None]
                ki=torch.arange(lk,device=device)[None]
                ends=((qi//num_frame_per_block)+1)*num_frame_per_block
                causal_masks[key]=~((ki<ends)|(qi==ki))[None,None]
            ca=causal_masks[key]
            blocked=ca if blocked is None else blocked|ca
        mask=None if blocked is None else ~blocked
        out=F.scaled_dot_product_attention(q.transpose(1,2).to(dtype),
              k.transpose(1,2).to(dtype),v.transpose(1,2).to(dtype),
              attn_mask=mask,dropout_p=dropout_p)
        return out.transpose(1,2).contiguous()
    return attention


def frozen_scale_forward(original, source):
    tree=ast.parse(Path(source).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CausalSelfAttention')
    node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    class Replace(ast.NodeTransformer):
        count=0
        def visit_Call(self,n):
            if ast.unparse(n)=='self.scale.item()':
                self.count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_literal_scale',ctx=ast.Load()),
                                                 args=[ast.Name(id='self',ctx=ast.Load())],keywords=[]),n)
            return self.generic_visit(n)
    transform=Replace();node=transform.visit(node);assert transform.count==1
    cache={}
    def literal_scale(mod):
        p=mod.scale
        assert not p.requires_grad, 'Only a frozen, originally detached constant is eligible'
        key=(id(mod),p.data_ptr(),p._version,str(p.device),p.dtype)
        if key not in cache:cache[key]=p.item()
        return cache[key]
    env=dict(original.__func__.__globals__);env['_literal_scale']=literal_scale
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 str(source)+':literal-scale-reuse','exec'),env)
    return env['forward']


def install(raw):
    if hasattr(raw,'_set_exact_core_acceleration'):return raw._set_exact_core_acceleration
    enabled=[False]
    def switch(value):enabled[0]=bool(value)
    if raw.method=='cmdm':
        env=raw.core.blocks[0].self_attn.forward.__func__.__globals__
        original=env['attention']
        env['attention']=cached_attention(original,enabled)
    elif raw.method=='motionstreamer':
        for block in raw.core.transformer.h:
            mod=block.attn;original=mod.forward
            candidate=frozen_scale_forward(original,raw.source)
            def forward(self,*args,_old=original,_new=candidate,**kw):
                if not enabled[0] or self.scale.requires_grad:return _old(*args,**kw)
                return _new(self,*args,**kw)
            mod.forward=types.MethodType(forward,mod)
    else:raise ValueError(raw.method)
    object.__setattr__(raw,'_set_exact_core_acceleration',switch)
    return switch
