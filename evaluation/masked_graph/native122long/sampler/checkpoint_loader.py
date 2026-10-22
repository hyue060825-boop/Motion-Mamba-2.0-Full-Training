"""Read-only loaders for the three trained 100k architectures."""
import os, sys, json, hashlib
from pathlib import Path
import torch

def load(args):
    root = Path(os.environ['EVAL_TRAINING_ROOT'])
    arch = os.environ['EVAL_ARCH']
    if arch in ('ardy', 'minimal'):
        import adapter
        model, client, checkpoint = adapter.load(args.checkpoint, False, device=args.device)
    else:
        from motion_training import runtime
        model, client, checkpoint = runtime.load(args.checkpoint, False, device=args.device)
    assert str(Path(args.stats_dir).resolve()) == str(Path(checkpoint['config']['stats_dir']).resolve())
    # Framework checkpoints store the completed global step, even when the filename is stage-local.
    global_step = int(checkpoint['step'])
    expected=json.loads((Path(__file__).parent/'CHECKPOINTS.json').read_text())[os.environ['EVAL_LABEL']]['step']
    assert global_step == expected, (arch, global_step, expected)
    assert model.gen_horizon_len == 52 and model.motion_rep.fps == 25
    assert model.motion_rep.motion_rep_dim == 414
    assert not checkpoint['config'].get('content_memory_enabled', False)
    if arch == 'ardy':
        client = model
        assert args.mode == 'original'
        sequence_architecture = None
    else:
        from motion_training._compat.checkpoint_config import sequence_config_kwargs
        _, sequence_architecture = sequence_config_kwargs(args, checkpoint)
        assert args.mode == ('minimal' if arch == 'minimal' else 'hybrid')
    raw = model.denoiser.model
    for key in ('root_model', 'body_model'):
        state = getattr(raw, key).state_dict()
        assert set(state) == set(checkpoint[key]), key
        assert all(torch.equal(v.detach().cpu(), checkpoint[key][k].to(v.dtype).cpu()) for k,v in state.items()), key
    assert all(torch.equal(raw.state_dict()[k].detach().cpu(), v.to(raw.state_dict()[k].dtype).cpu()) for k,v in checkpoint['denoiser_aux'].items())
    assert not any(p.requires_grad for p in model.parameters())
    out = Path(args.output).parent
    metadata = dict(architecture=arch, checkpoint=args.checkpoint, global_step=global_step,
        local_step=checkpoint['step'], framework_global_start=checkpoint.get('_framework_global_start',0),
        training_root=str(root), stats_dir=args.stats_dir,
        generation_contract=getattr(model,'generation_contract',None),
        decoder_history_frames=getattr(model,'decoder_history_frames',None),
        padding_contract=checkpoint['config'].get('padding_contract'),
        all_loaded_denoiser_weights_verified=True, persistent_content_memory=False,
        inference_backend='reference; no unverified kernel substitutions',
        config=checkpoint['config'], initialization=checkpoint.get('initialization'),
        parameter_count=sum(p.numel() for p in raw.parameters()))
    (out/'LOAD.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,default=str)+'\n')
    return model, client, checkpoint, sequence_architecture
