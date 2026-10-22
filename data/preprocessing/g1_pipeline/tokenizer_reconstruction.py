"""Actual frozen G1 tokenizer reconstruction, through the official hybrid path."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .common import ROOT,read_jsonl,sha,write_json,setup_imports
from .gates import require_pass
from .tokenizer_normalizer_bridge import FrozenTokenizerBridge


def run(namespace,device='cpu'):
    require_pass('03_hiphi100');setup_imports()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from motion_mamba2.representation import MotionMamba2MotionRep
    from motion_mamba2.models.latent_utils import HybridMotionConverter
    assets=json.loads((ROOT/'audit/training_assets_v1.json').read_text());snapshot=Path(assets['models']['g1']['snapshot'])
    for name,bound in assets['models']['g1']['file_hashes'].items():
        if sha(snapshot/name)!=bound:raise ValueError('TOKENIZER_CHECKPOINT_CHANGED')
    conf=OmegaConf.load(snapshot/'config.yaml');conf.checkpoint_dir=str(snapshot)
    auto=instantiate(OmegaConf.create(OmegaConf.to_container(conf.autoencoder,resolve=True))).to(device).eval()
    released=auto.motion_rep;contract=json.loads((ROOT/'contracts/motion_mamba2_414.json').read_text())
    project=MotionMamba2MotionRep(released.skeleton,25,stats_path=contract['chosen_normalizer']);bridge=FrozenTokenizerBridge(auto,project)
    raw=SimpleNamespace(nframe_root_dim=20)
    official=HybridMotionConverter(denoiser=raw,autoencoder=auto,motion_rep=released,gen_horizon_len=52)
    adapted=HybridMotionConverter(denoiser=raw,autoencoder=bridge,motion_rep=project,gen_horizon_len=52)
    panelpath=ROOT/'work'/namespace/'sim_panel.jsonl';panel=read_jsonl(panelpath);results=[]
    for row in panel:
        feature=ROOT/row['output_paths']['features_path'];physical=np.load(feature,allow_pickle=False)
        if physical.ndim!=2 or physical.shape[1]!=414 or not np.isfinite(physical).all():raise ValueError('TOKENIZER_INPUT_NOT_G1')
        # Fixed 104-frame chunks cover every real frame once; terminal padding
        # is for transformer patch shape and is excluded from every metric.
        errors=[];featureerrors=[];bridgeerrors=[];pos_consistency=[];contacts=[]
        with torch.inference_mode():
            for start in range(0,len(physical),104):
                n=min(104,len(physical)-start);x=torch.tensor(physical[start:start+n],device=device).float()[None]
                padded=torch.cat([x,x[:,-1:].expand(1,104-n,414)],dim=1) if n<104 else x
                length=torch.tensor([n],device=device);mask=torch.arange(104,device=device)[None]<length[:,None]
                normalized=released.normalize(padded);token,_=official.get_hybrid_motion_from_explicit(normalized,length,mask)
                reconstructed=released.unnormalize(official.get_explicit_motion_from_hybrid(token,mask,length))
                projectnormalized=project.normalize(padded);projecttoken,_=adapted.get_hybrid_motion_from_explicit(projectnormalized,length,mask)
                adapted_reconstruction=project.unnormalize(adapted.get_explicit_motion_from_hybrid(projecttoken,mask,length))
                expected=released.inverse(padded,is_normalized=False)['posed_joints'][:,:n]
                actual=released.inverse(reconstructed,is_normalized=False)['posed_joints'][:,:n]
                errors.append(torch.linalg.vector_norm(actual-expected,dim=-1).cpu().numpy().reshape(-1,34))
                featureerrors.append(abs(reconstructed[:,:n]-padded[:,:n]).cpu().numpy().reshape(-1,414))
                bridgeerrors.append(float(abs(adapted_reconstruction[:,:n]-reconstructed[:,:n]).max()))
                # Decoder positions and rotation-FK are redundant predictions;
                # report inconsistency rather than claiming every generated
                # feature block is constrained by the deterministic encoder.
                generated_pos=reconstructed[:,:n,5:104].reshape(1,n,33,3).clone()
                generated_pos[...,0]+=reconstructed[:,:n,None,0];generated_pos[...,2]+=reconstructed[:,:n,None,2]
                pos_consistency.append(torch.linalg.vector_norm(generated_pos-actual[:,:,1:],dim=-1).cpu().numpy().ravel())
                contacts.append(float(abs(reconstructed[:,:n,410:414]-padded[:,:n,410:414]).mean()))
        e=np.concatenate(errors);f=np.concatenate(featureerrors);body_names=released.skeleton.bone_order_names
        wrists=[i for i,name in enumerate(body_names) if name in ('left_wrist_yaw_skel','right_wrist_yaw_skel')]
        feet=list(released.skeleton.foot_joint_idx)
        result={'sample_id':row['sample_id'],'dataset':row['dataset'],'frames':len(physical),'matching_class':row['matching_class'],
            'fk_mpjpe_m':float(e.mean()),'fk_error_p95_m':float(np.percentile(e,95)),'per_joint_mpjpe_m':e.mean(axis=0).tolist(),
            'feet_mean_m':float(e[:,feet].mean()),'wrists_mean_m':float(e[:,wrists].mean()) if len(wrists)==2 else None,
            'feature_block_mae':{'root':float(f[:,:5].mean()),'positions':float(f[:,5:104].mean()),'rotation6d':float(f[:,104:308].mean()),
                'velocities':float(f[:,308:410].mean()),'contacts':float(f[:,410:].mean())},
            'decoded_position_vs_rotation_fk_mean_m':float(np.concatenate(pos_consistency).mean()),
            'bridge_vs_released_reconstruction_max_abs':max(bridgeerrors),'feature_sha256':sha(feature),
            'note':'learned reconstruction errors are not deterministic 414 encode/inverse contract errors'}
        results.append(result);print(row['sample_id'],result['fk_mpjpe_m'],'reconstruction',flush=True)
    report={'status':'MEASURED_PENDING_FROZEN_FRONTEND_ACCEPTANCE','namespace':namespace,'results':results,
        'panel_sha256':sha(panelpath),'asset_audit_sha256':sha(ROOT/'audit/training_assets_v1.json'),
        'module_sha256':sha(__file__),'bridge_module_sha256':sha(ROOT/'g1_pipeline/tokenizer_normalizer_bridge.py'),
        'normalizer_selection':'project external normalizer unchanged; frozen tokenizer internally retains released stats',
        'valid_frame_policy':'104-frame transformer chunks; explicit real lengths and masks; no padding in reported errors',
        'systematic_compatibility_pass':None,'systematic_compatibility_reason':'requires per-source/class baseline comparison and explicit reviewed reconstruction limits',
        'training_run':False}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_tokenizer_reconstruction.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--namespace',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();run(a.namespace,a.device)
