"""First-order CUDA Graph of frozen B1 geometry; denoiser and Adam stay eager."""
import contextlib
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import pipeline_accel as fast
from motion_training import losses, batching
from g1_motion_contract import decode_training_prediction, CONTRACT
from g1_training_precision import prediction_loss_context

VERSION = 'frozen_B1_geometry_forward_backward_graph_v1'


@contextlib.contextmanager
def explicit_decoder_causality(converter):
    decoder = converter.autoencoder.decoder
    module = decoder.seqTransEncoder
    original = module.forward
    def forward(*args, **kwargs):
        kwargs['is_causal'] = bool(decoder.is_causal)
        return original(*args, **kwargs)
    module.forward = forward
    try:
        yield
    finally:
        module.forward = original


def fixed_G_components(raw, converter, pred, fields):
    """Same B1 operations, with already selected, contiguous generation rows."""
    total, diagnostic = [], []
    one = torch.ones(1, 13, 1, dtype=torch.bool, device=pred.device)
    valid = torch.ones(1, 52, dtype=torch.bool, device=pred.device)
    for i in range(pred.shape[0]):
        p = pred[i:i+1]
        values = F.smooth_l1_loss(p.float(), fields['target'][i:i+1].float(), beta=1, reduction='none')
        hybrid, _ = losses.mean_valid(values, one)
        root, _ = losses.mean_valid(values[:, :, :20], one)
        body, _ = losses.mean_valid(values[:, :, 20:], one)
        prefix = fields.get('prefix')
        dec = decode_training_prediction(p.float(), dict(motion_contract=CONTRACT,
            hybrid_converter=converter, history_mask=~valid, generation_mask=valid,
            decoder_history_hybrid=None if prefix is None else prefix[i:i+1]))
        decoded = (dec[:, :, raw.motion_rep.body_slice].float()-fields['supervised'][i:i+1, :, raw.motion_rep.body_slice].float()).abs().flatten(1).mean(1).mean()
        goal, _ = losses.mean_valid((dec.float()-fields['observed'][i:i+1].float()).abs(), fields['mask'][i:i+1])
        fk = raw.motion_rep.inverse(dec, is_normalized=True)['posed_joints'][:, :, 1:].float()
        positions = raw.motion_rep.inverse(dec, is_normalized=True, posed_joints_from='positions')['posed_joints'][:, :, 1:].float()
        consistency = (fk-positions).norm(dim=-1).flatten(1).mean(1).mean()
        total.append(hybrid+decoded+goal+consistency)
        diagnostic.append(torch.stack([hybrid.detach(), decoded.detach(), goal.detach(), consistency.detach(), root.detach(), body.detach()]))
    return torch.stack(total).mean(), torch.stack(diagnostic).mean(0)


class GeometryGraphs:
    def __init__(self, raw, converter):
        self.raw, self.converter = raw, converter
        self.entries = {}
        self.replays = 0

    def run(self, pred, fields):
        assert pred.dtype == torch.float32 and pred.shape[1:] == (13, 148)
        assert not self.converter.autoencoder.training
        assert not any(p.requires_grad for p in self.converter.autoencoder.parameters())
        prefix = fields.get('prefix')
        key = (pred.shape[0], None if prefix is None else tuple(prefix.shape[1:]), pred.device.index)
        if key not in self.entries:
            for rep in [self.raw.motion_rep, self.converter.motion_rep]:
                for name in ['global_root_stats', 'local_root_stats', 'body_stats']:
                    getattr(rep, name).to(device=pred.device)
            static_pred = pred.detach().clone().requires_grad_(True)
            static = {k:v.detach().clone() if v is not None else None for k,v in fields.items()}
            stream = torch.cuda.Stream(device=pred.device)
            stream.wait_stream(torch.cuda.current_stream(pred.device))
            with torch.cuda.stream(stream), torch.enable_grad(), prediction_loss_context(self.raw, static_pred), explicit_decoder_causality(self.converter):
                for _ in range(2):
                    if static_pred.grad is not None:
                        static_pred.grad.zero_()
                    value, diag = fixed_G_components(self.raw, self.converter, static_pred, static)
                    value.backward()
            torch.cuda.current_stream(pred.device).wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.enable_grad(), prediction_loss_context(self.raw, static_pred), explicit_decoder_causality(self.converter):
                with torch.cuda.graph(graph, stream=stream):
                    static_pred.grad.zero_()
                    value, diag = fixed_G_components(self.raw, self.converter, static_pred, static)
                    value.backward()
            self.entries[key] = (static_pred, static, graph, value, diag)
        static_pred, static, graph, value, diag = self.entries[key]
        static_pred.detach().copy_(pred.detach())
        for k,v in fields.items():
            if v is not None:
                static[k].copy_(v)
        graph.replay()
        self.replays += 1
        return value.detach().clone(), diag.detach().clone(), static_pred.grad.detach().clone()


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, cache, fields):
        value, diag, gradient = cache.run(pred, fields)
        ctx.save_for_backward(gradient)
        ctx.mark_non_differentiable(diag)
        return value, diag

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_value, grad_diag):
        (gradient,) = ctx.saved_tensors
        return gradient*grad_value, None, None


class LossModule(torch.nn.Module):
    def __init__(self, raw, client):
        super().__init__()
        self.raw, self.client = raw, client
        self.cache = None

    def forward(self, fields):
        self.client.reset_memory()
        pred = self.raw(*(fields[k] for k in losses.FORWARD_KEYS))
        h = int(fields['history_len'][0])
        assert torch.equal(fields['history_len'], fields['history_len'][:1].expand_as(fields['history_len']))
        lo = h//4
        p = pred[:, lo:lo+13]
        fixed = dict(target=fields['target'][:, lo:lo+13], supervised=fields['supervised_motion'],
                     observed=fields['observed'][:, h:h+52], mask=fields['motion_mask'][:, h:h+52].bool(),
                     prefix=fields.get('decoder_history_hybrid'))
        if self.cache is None:
            self.cache = GeometryGraphs(self.raw, fields['hybrid_converter'])
        value, diag = _Geometry.apply(p, self.cache, fixed)
        return value, dict(diagnostics=diag)


def install():
    update = fast.install('serial_codec', 4)
    batching.LossModule = LossModule
    return update
