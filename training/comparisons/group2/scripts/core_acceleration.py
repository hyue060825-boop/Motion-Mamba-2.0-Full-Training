"""Exact positional arithmetic reuse; no learned state, inputs, or precision changes."""
import torch

def install(raw):
    if getattr(raw,'_exact_core_acceleration',False):return raw._set_exact_core_acceleration
    enabled=[False]
    def switch(value):enabled[0]=bool(value)
    if raw.method=='fd2':
        env=raw.core.blocks[0].self_attn.forward.__func__.__globals__
        original=env['rope_apply']
        @torch.amp.autocast('cuda',enabled=False)
        def rope(x,grid_sizes,freqs,channel_split=None,token_lens=None,position_ids=None):
            # Immutable CPU shape metadata only. Unsupported cases use original.
            grids=grid_sizes.tolist() if grid_sizes.device.type=='cpu' else None
            if enabled[0] and channel_split==(1,0,0) and token_lens is None and position_ids is None and grids is not None and len(grids)==len(x) and all(tuple(g)==(x.shape[1],1,1) for g in grids):
                b,l,h,d=x.shape
                z=torch.view_as_complex(x.to(torch.float64).reshape(b,l,h,d//2,2))
                return torch.view_as_real(z*freqs[:l].view(1,l,1,d//2)).flatten(-2).float()
            return original(x,grid_sizes,freqs,channel_split,token_lens,position_ids)
        env['rope_apply']=rope
    elif raw.method=='omg':
        # Share exact official immutable sin/cos across identical heads/layers.
        cache={}
        for layer in raw.core.layers:
            mod=layer.self_attn.rope
            original=mod.forward
            frequency_signature=tuple(mod.inv_freq.detach().cpu().tolist())
            def cached(seq_len,device,dtype,_mod=mod,_original=original,_signature=frequency_signature):
                if not enabled[0]:return _original(seq_len,device,dtype)
                key=(int(seq_len),str(device),dtype,_mod.inv_freq._version,str(_mod.inv_freq.device),_mod.inv_freq.dtype,_signature)
                if key not in cache:cache[key]=_original(seq_len,device,dtype)
                return cache[key]
            mod.forward=cached
    else:raise ValueError(raw.method)
    object.__setattr__(raw,'_exact_core_acceleration',True)
    object.__setattr__(raw,'_set_exact_core_acceleration',switch)
    return switch
