"""Read-only cached-model/normalizer provenance; never starts training."""
from motion_mamba2.paths import artifact_path
import json
from pathlib import Path
import numpy as np
import yaml
from g1_pipeline.common import ROOT,REPO,sha,write_json


def run():
    cache=Path(artifact_path('hub_27e995e390'));models={}
    revisions={
        'g1':'models--nvidia--ARDY-G1-RP-25FPS-Horizon52/snapshots/059b8007df0ba194a006a877b59a563955ac7b70',
        'llama':'models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',
        'mntp':'models--McGill-NLP--LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/snapshots/31474e395ada192e8ed1586db6be79fb3b70c9c0',
        'supervised':'models--McGill-NLP--LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/snapshots/baa8ebf04a1c2500e61288e7dad65e8ae42601a7'}
    for name,relative in revisions.items():
        root=cache/relative
        files=sorted(p for p in root.rglob('*') if p.is_file())
        models[name]={'snapshot':str(root),'bytes':sum(p.stat().st_size for p in files),
            'file_hashes':{str(p.relative_to(root)):sha(p) for p in files}}
        print(name,'bound',len(files),flush=True)
    contract=json.loads((ROOT/'contracts/motion_mamba2_414.json').read_text());chosen=Path(contract['chosen_normalizer'])
    comparisons={};released=cache/revisions['g1']/'stats/motion'
    for name in ('mean.npy','std.npy'):
        a=np.load(chosen/name);b=np.load(released/name)
        comparisons[name]={'project_shape':list(a.shape),'released_shape':list(b.shape),'project_sha256':sha(chosen/name),
            'released_sha256':sha(released/name),'max_absolute_difference':float(abs(a-b).max()),'equal':bool(np.array_equal(a,b))}
    cfg=yaml.safe_load((cache/revisions['g1']/'config.yaml').read_text())
    report={'status':'TRAINING_ASSET_METADATA_AUDIT_ONLY','models':models,'normalizers':comparisons,
        'model_contract':{'skeleton':cfg['autoencoder']['motion_rep']['skeleton']['_target_'],'fps':cfg['autoencoder']['motion_rep']['fps'],
            'frames_per_token':cfg['autoencoder']['num_frames_per_token'],'horizon':cfg['gen_horizon_len'],'text_width':cfg['denoiser']['llm_shape'][-1]},
        'project_normalizer_sha256':{'mean':sha(chosen/'mean.npy'),'std':sha(chosen/'std.npy')},
        'compatibility_status':'PENDING_MEASURED_TOKENIZER_RECONSTRUCTION',
        'compatibility_reason':'Pinned project BONES stats differ from released frozen tokenizer stats; loader roundtrip alone cannot validate frontend compatibility',
        'normalizer_changed':False,'training_run':False,'source_retargeting_run':False,
        'training_helpers':{str(p):sha(p) for p in [REPO/'mamba-transformer-keneral/examples/train_humanml3d_motion_mamba2_tiny.py',
            REPO/'mamba-transformer-keneral/src/motion_mamba2.adapters/text_encoder.py',REPO/'mamba-transformer-keneral/src/motion_mamba2.adapters/integration.py']},
        'hf_download_endpoint':'https://hf-mirror.com','download_required':False}
    write_json(ROOT/'audit/training_assets_v1.json',report)
    return report


if __name__=='__main__':print(run()['compatibility_status'])
