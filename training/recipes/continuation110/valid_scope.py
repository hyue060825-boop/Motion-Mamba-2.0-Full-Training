"""Factor all-valid checks within canonical no-grad accepted-history replay.

This is the existing attention's validated-all-valid switch, scoped only to
one canonical chunk. Cache identity/projection-version guards and all commits
still execute. Changed or previously unseen validity tensors are checked on
GPU; a validity proof is never reused after its mutation counter changes.
Current-window forward/backward and online control are never in this scope.
"""
import collections
import torch
import history_v3

_enabled=False;_original=None;_known={};_stats=collections.Counter()

def begin(enabled):
    global _enabled
    _enabled=bool(enabled);_known.clear();_stats.clear()

def install():
    global _original
    if _original is not None:return
    _original=history_v3.raw_history_forward
    history_v3.raw_history_forward=forward

def signature(t):
    return (id(t),t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),t.device,t.dtype)

def prove(tensors):
    pending=[t for t in tensors if signature(t) not in _known]
    if pending:
        assert all(t.dtype==torch.bool for t in pending)
        # A single synchronized boolean read validates all unseen masks. No
        # false mask is inferred to be true from shape, source name or padding.
        assert bool(torch.cat([t.reshape(-1) for t in pending]).all()),'Noncanonical invalid history mask'
        for t in pending:_known[signature(t)]=t
        _stats['combined_validations']+=1;_stats['validated_tensors']+=len(pending)

def caches(raw):
    for block in [raw.root_model,raw.body_model]:
        state=getattr(block,'_sequence_state',None)
        if state is not None:
            for layer in state.layers:
                yield layer.attention_cache

def forward(raw,tokens,layout,kw):
    if not _enabled:return _original(raw,tokens,layout,kw)
    assert not raw.training and not torch.is_grad_enabled()
    assert kw['include_synthetic_generation'] is False and kw.get('detach',True)
    # Current history query validity and all existing cache validity are checked
    # before activating the attention fast path. First state is empty; an
    # unfamiliar cloned prefix undergoes the same numeric validation.
    prove([layout.history_token_mask]+[chunk.token_valid for cache in caches(raw) for chunk in cache.chunks])
    attentions=[layer.mixer.attention for block in [raw.root_model,raw.body_model] for layer in block.sequence_stack.layers]
    previous=[getattr(attention,'_validated_graph_all_valid',False) for attention in attentions]
    for attention in attentions:attention._validated_graph_all_valid=True
    try:
        result=_original(raw,tokens,layout,kw)
        _stats['canonical_chunks']+=1
        # A new pending current_mask is created by the original prepare as
        # torch.ones when current_mask=None. We still numerically validate new
        # committed masks together at the next chunk, rather than assuming the
        # cache writer cannot change them. No write/eviction is skipped.
        return result
    finally:
        for attention,value in zip(attentions,previous):attention._validated_graph_all_valid=value

def end():
    global _enabled
    _known.clear();_enabled=False

def stats():return dict(_stats,known_mask_tensors=len(_known),scope='canonical no-grad history only')
