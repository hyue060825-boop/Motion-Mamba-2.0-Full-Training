"""Pinned CMDM / MotionStreamer causal cores in the shared G1 experiment.

These are architecture adaptations. The shared codec, x0 target, sampler and
four losses replace the papers' original codec and training/sampling schedules.
Spatial observations use each core's existing condition route; no extra labels,
future GT, motion planner, semantic head or Mamba state is introduced.
"""
from pathlib import Path
from types import SimpleNamespace
import ast, math, hashlib, typing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class FP32Linear(nn.Linear):
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.weight, self.bias)


def definitions(path, env, names=None):
    """Load pinned compute definitions without heavyweight data/train imports."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and (names is None or n.name in names)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), env)
    return env


class ExternalDenoiser(nn.Module):
    def __init__(self, method, repo, motion_rep, hybrid):
        super().__init__()
        self.method = method
        object.__setattr__(self, 'motion_rep', motion_rep)
        object.__setattr__(self, 'hybrid', hybrid)
        self.nframe_root_dim = 20
        self.latent_embedding_dim = 128
        self.training_precision_contract = 'g1_root_output_geometry_fp32_v1'
        self.inactive_prefixes = []
        env = dict(torch=torch, nn=nn, F=F, math=math, np=np, rearrange=rearrange,
                   Optional=typing.Optional, Self=typing.Any)
        if method == 'cmdm':
            helper = Path(repo)/'models/Transformer.py'
            definitions(helper, env)
            source = Path(repo)/'models/Causal_DiT.py'
            # Original transport, text encoder and dataset frontend are outside
            # the matched-G1 denoiser comparison and must not be loaded here.
            env.update(create_transport=lambda **kw: None, Sampler=lambda _: None)
            definitions(source, env)
            base = env['DiT']
            core_type = type('G1CMDMCore', (base,), {'load_and_freeze_clip': lambda self: nn.Identity()})
            self.core = core_type(input_dim=3465, cond_mode='text', latent_dim=512,
                                  ff_size=1024, num_layers=8, num_heads=4, dropout=0,
                                  clip_dim=4096, cond_mask_prob=0., num_frame_per_block=3,
                                  is_causal=True, max_length=1024)
            self.core.final_layer = FP32Linear(512, 148)
            nn.init.zeros_(self.core.final_layer.weight); nn.init.zeros_(self.core.final_layer.bias)
            self.source_edits = [
                'CMDM published dit() configuration:512 hidden,8 causal blocks,4 heads,FF1024,3-token causal blocks; actual official block/forward_step arithmetic',
                'common G1 hybrid148 and observations/masks replace MAC-VAE latent representation; official x/text projection input widths adapted',
                'allowed sparse observations plus timestamps enter existing cross-attention context, visible from first generation window; causal motion self-attention unchanged',
                'cached4096 text replaces DistilBERT frontend; common data pipeline applies dropout once; internal duplicate condition dropout disabled',
                'common x0/noise/four losses replace original flow/pyramid training and sampling; not full official CMDM system',
                'nonlearned RoPE table reserves1024 tokens; no temporal compression or learned positional extrapolator']
            self.helper_sources = [str(helper)]
        elif method == 'motionstreamer':
            source = Path(repo)/'models/llama_model.py'
            env['LLaMAHFConfig'] = SimpleNamespace
            definitions(source, env, {'Block', 'CausalSelfAttention', 'LengthCausalSelfAttention',
                                      'MLP', 'RMSNorm', 'build_rope_cache', 'apply_rope'})
            diff = Path(repo)/'models/diffloss.py'
            # Keep the original causal Transformer AND its adaptive diffusion
            # MLP. Only its noise/loss scheduler is supplied by the shared recipe.
            definitions(diff, env, {'modulate', 'TimestepEmbedder', 'ResBlock', 'FinalLayer', 'SimpleMLPAdaLN'})
            cfg = SimpleNamespace(block_size=78, n_layer=12, n_head=12, n_embd=768, T5_xxl_dim=4096)
            self.core = nn.Module()
            self.core.transformer = nn.ModuleDict(dict(
                wte=nn.Linear(3465, 768), cond_embed=nn.Linear(4096, 768),
                h=nn.ModuleList([env['Block'](cfg) for _ in range(12)]), ln_f=env['RMSNorm'](768)))
            self.core.out_proj = nn.Linear(768, 768)
            self.core.diffusion_head = env['SimpleMLPAdaLN'](
                in_channels=148, model_channels=1792, out_channels=148,
                z_channels=768, num_res_blocks=9, grad_checkpointing=False)
            final = self.core.diffusion_head.final_layer.linear
            fp32 = FP32Linear(final.in_features, final.out_features)
            fp32.load_state_dict(final.state_dict()); self.core.diffusion_head.final_layer.linear = fp32
            for i, block in enumerate(self.core.transformer.h):
                # The official forward calls scale.item(): it is a detached
                # constant, not a learnable parameter. Retain the78-position
                # initialization value even when allocating a larger RoPE table.
                block.attn.scale.requires_grad_(False)
                self.inactive_prefixes.append(f'core.transformer.h.{i}.attn.scale')
                block.attn.block_size = 1024
            self.source_edits = [
                'MotionStreamer Normal_size:12 LLaMA blocks,768 hidden,12 heads,QK normalization,RoPE,RMSNorm,SwiGLU retained',
                'original adaptive diffusion MLP head retained:1792 width,9 residual blocks; common hybrid148 input/output replaces original16D TAE latent',
                'common G1 codec and cached4096 text replace original CausalTAE/sentenceT5 assets; original two-forward semantic-stream training is not claimed',
                'numeric sparse observations/masks and timestamps use existing condition-prefix projection before causal motion tokens, with no extra GT path',
                'motion context shifted by one latent token; previous noisy G context follows common noising recipe, never clean target-G teacher forcing',
                'common x0/noise/four losses and52-frame generation replace original per-token epsilon sampler/two-forward training; not full official MotionStreamer',
                'detached official attention scale retained as fixed constant; nonlearned RoPE table enlarged to1024 positions while retaining original78-scale value']
            self.helper_sources = [str(diff)]
        else:
            raise ValueError(method)
        self.source = str(source)
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.apply_trainable_policy()
        # Admitted literal metadata reuse; inputs, core math and parameters unchanged.
        import os,json
        run_root=os.environ.get('BASELINE_RUN_ROOT')
        receipt=Path(run_root)/'gates/CORE_ACCELERATION.json' if run_root else None
        if receipt is not None and receipt.exists() and json.loads(receipt.read_text()).get('admit',False):
            from core_acceleration import install
            install(self)(True)

    def apply_trainable_policy(self):
        for name, p in self.named_parameters():
            p.requires_grad_(not any(name.startswith(k) for k in self.inactive_prefixes))

    def prepared(self, x, history_len, generation_len, future_len, history_mask,
                 generation_mask, future_mask, history_token_mask, generation_token_mask,
                 future_token_mask, text_feat, text_pad_mask, t, first_heading_angle,
                 motion_mask=None, observed_motion=None):
        b, n, d = x.shape
        assert d == 148 and bool((generation_len == 52).all())
        assert observed_motion.shape == motion_mask.shape == (b, n*4, 414)
        mask = motion_mask.float(); obs = observed_motion.float()*mask
        role = torch.stack([history_token_mask, generation_token_mask, future_token_mask], -1).float()
        valid = role.bool().any(-1)
        angle = torch.stack([first_heading_angle.sin(), first_heading_angle.cos()], -1)
        packed = torch.cat([x.float(), torch.cat([obs, mask], -1).reshape(b, n, 3312),
                            role, angle[:, None].expand(-1, n, -1)], -1)
        text = text_feat.float()[:, 0]*text_pad_mask[:, 0, None]
        return packed, text, valid

    def forward(self, *args, **kwargs):
        keys = ['x','history_len','generation_len','future_len','history_mask','generation_mask',
                'future_mask','history_token_mask','generation_token_mask','future_token_mask',
                'text_feat','text_pad_mask','t','first_heading_angle','motion_mask','observed_motion']
        bound = dict(zip(keys, args)); bound.update(kwargs)
        if 'timesteps' in bound: bound['t'] = bound.pop('timesteps')
        if 'text_feat_pad_mask' in bound: bound['text_pad_mask'] = bound.pop('text_feat_pad_mask')
        a, text, valid = self.prepared(**bound)
        b, n, _ = a.shape
        assert n <= 512, 'Actual paired40s input fits; no cropping or target relocation'
        # Condition tokens contain only allowed observations, masks, roles,
        # heading and timestamps. Unknown numerical coordinates are zeroed.
        stamp = torch.arange(n, device=a.device, dtype=a.dtype)[None, :, None].expand(b, -1, -1)*(.16)
        spatial = F.pad(torch.cat([a[:, :, 148:], stamp], -1), (0, 4096-3318))
        if self.method == 'cmdm':
            cond = torch.cat([text[:, None], spatial], 1).transpose(0, 1)
            cond_mask = torch.ones(b, n+1, device=a.device, dtype=torch.bool)
            return self.core.forward_step(a, bound['t'], cond, cond_mask, valid)[:, :, 0]
        prefix = self.core.transformer.cond_embed(torch.cat([text[:, None], spatial], 1))
        motion = self.core.transformer.wte(a[:, :-1])
        z = torch.cat([prefix, motion], 1)
        for block in self.core.transformer.h: z = block(z)
        # Predict token i using only condition tokens and motion tokens <i.
        z = self.core.out_proj(self.core.transformer.ln_f(z[:, n:n+n]))
        t = bound['t']
        if t.ndim == 1: t = t[:, None].expand(b, n)
        pred = self.core.diffusion_head(a[:, :, :148].reshape(b*n, 148),
                                        t.reshape(b*n), z.reshape(b*n, 768))
        return pred.reshape(b, n, 148)
