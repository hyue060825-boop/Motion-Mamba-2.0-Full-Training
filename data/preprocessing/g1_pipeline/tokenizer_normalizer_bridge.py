"""Explicit normalization bridge; frozen tokenizer retains its checkpoint stats.

This optional frontend must pass actual reconstruction validation before use.
It does not modify model assets, dataset stats, or the physical 414-D format.
"""
import torch


class FrozenTokenizerBridge(torch.nn.Module):
    def __init__(self,tokenizer,project_rep):
        super().__init__();self.inner=tokenizer;self.motion_rep=project_rep
        released=tokenizer.motion_rep
        if project_rep.skeleton.nbjoints!=34 or released.skeleton.nbjoints!=34 or project_rep.fps!=25 or released.fps!=25:
            raise ValueError('TOKENIZER_BRIDGE_REQUIRES_G1_34_AT_25HZ')
        if project_rep.body_dim!=409 or released.body_dim!=409 or project_rep.local_root_dim!=4 or released.local_root_dim!=4:
            raise ValueError('TOKENIZER_BRIDGE_FEATURE_SEMANTICS')
        self.inner.eval()
        for p in self.inner.parameters():p.requires_grad_(False)

    @property
    def num_frames_per_token(self):return self.inner.num_frames_per_token
    @property
    def encode_with_quantization(self):return self.inner.encode_with_quantization
    @property
    def encode_with_normalization(self):return self.inner.encode_with_normalization

    def tokenize(self,x,motion_pad_mask):
        physical=self.motion_rep.unnormalize(x)
        released=self.inner.motion_rep.normalize(physical)
        return self.inner.tokenize(released,motion_pad_mask)

    def detokenize(self,token_embeddings,external_cond=None,motion_pad_mask=None):
        if external_cond is not None:
            physical=self.motion_rep.local_root_stats.unnormalize(external_cond)
            external_cond=self.inner.motion_rep.local_root_stats.normalize(physical)
        decoded=self.inner.detokenize(token_embeddings,external_cond=external_cond,motion_pad_mask=motion_pad_mask)
        result={}
        for key,value in decoded.items():
            if key=='body':origin=self.inner.motion_rep.body_stats;dest=self.motion_rep.body_stats
            elif key=='root':origin=self.inner.motion_rep.local_root_stats;dest=self.motion_rep.local_root_stats
            else:raise ValueError('UNKNOWN_TOKENIZER_OUTPUT_BLOCK')
            result[key]=dest.normalize(origin.unnormalize(value))
        return result

    def requantize(self,token_embeddings):return self.inner.requantize(token_embeddings)
    def forward(self,*args,**kwargs):return self.detokenize(*args,**kwargs)
