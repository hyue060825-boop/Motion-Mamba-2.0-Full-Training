"""Released model loader: no training adapter and no stats replacement."""
from pathlib import Path
import os,sys,json,time
import torch
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from util import ROOT,read,write,sha
from official_setup import activate,OFFICIAL

def load(args):
    model=activate()('g152',device=args.device,text_encoder=False,checkpoints_dir=str(ROOT/'models'))
    info=read(ROOT/'OFFICIAL_RELEASE.json');snapshot=Path(info['snapshot'])
    for n,st in info['file_stats'].items():
        p=snapshot/n;s=p.stat();assert dict(size=s.st_size,mtime_ns=s.st_mtime_ns)==st,n
    for n,h in info['official_sources'].items():assert sha(OFFICIAL/n)==h,n
    assert Path(args.stats_dir).resolve()==(snapshot/'stats/motion').resolve()
    import motion_mamba2.models.motion_model as module
    assert Path(module.__file__).resolve().is_relative_to(OFFICIAL)
    assert sha(Path(module.__file__))==info['official_sources']['ardy/model/motion_mamba2_model.py']
    ds={k.removeprefix('denoiser.').removeprefix('backbone.'):v for k,v in load_file(str(snapshot/'denoiser.safetensors')).items() if k.startswith('denoiser.')}
    ts={k.removeprefix('pose_net.'):v for k,v in load_file(str(snapshot/'tokenizer.safetensors')).items() if k.startswith('pose_net.')}
    for name,state,ref in [('denoiser',model.denoiser.model.state_dict(),ds),('tokenizer',model.autoencoder.state_dict(),ts)]:
        assert set(state)==set(ref),(name,set(state)^set(ref))
        assert all(torch.equal(v.detach().cpu(),ref[k].to(v.dtype)) for k,v in state.items()),name
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert model.gen_horizon_len==52 and model.motion_rep.fps==25 and model.motion_rep.motion_rep_dim==414
    from official_rolling import attach,HISTORY_FRAMES
    model=attach(model,Path(args.output).parent/'INFERENCE_TRACE.json')
    metadata=dict(architecture='motion_mamba2_official_released',checkpoint=str(snapshot),global_step=None,
        official_repo=info['hf_repo'],official_weight_revision=info['cached_revision'],latest_revision=info['latest_revision'],
        same_weights_as_latest=info['weights_match_latest'],official_code_commit=info['code_commit'],
        all_loaded_denoiser_weights_verified=True,all_loaded_tokenizer_weights_verified=True,
        state_tensors_verified=dict(denoiser=len(ds),tokenizer=len(ts)),stats_dir=args.stats_dir,
        stats_replaced=False,training_adapter_used=False,postprocessing=False,steps=args.diffusion_steps,
        cfg=[args.cfg_weight_text,args.cfg_weight_constraint],inference_backend='official PyTorch reference',
        initial_history_frames=24,maximum_history_frames=HISTORY_FRAMES,
        history_and_future_policy='official autoregressive_step; CLI recommended196-frame history; all benchmark future goals preserved',
        benchmark_adapter_sha256=sha(ROOT/'official_rolling.py'),
        callable_proxy_used=True, assigned_forward_dispatch_not_used=True,
        official_model_module=module.__file__,time=time.time())
    write(Path(args.output).parent/'LOAD.json',metadata)
    return model,model,dict(mode='original'),None
