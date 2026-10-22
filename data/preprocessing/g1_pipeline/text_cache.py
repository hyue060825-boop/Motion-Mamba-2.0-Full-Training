"""Exact local ARDY text embeddings, deduplicated by text and encoder identity."""
import json
from pathlib import Path
import numpy as np
from .common import ROOT,read_jsonl,write_json,sha,digest,setup_imports
from .gates import require_pass
from .production_contract import save_array


def run(manifest=None,*,device='cuda:0',resume=True):
    require_pass('05_full_batch');setup_imports()
    from motion_mamba2.adapters.text_encoder import ExactMotionMamba2TextEncoder
    manifest=manifest or ROOT/'manifests/accepted_train.jsonl';records=read_jsonl(manifest)
    auditpath=ROOT/'audit/training_assets_v1.json';assets=json.loads(auditpath.read_text());modelroles=('llama','mntp','supervised')
    identity={k:assets['models'][k] for k in modelroles}
    for item in identity.values():
        for name,bound in item['file_hashes'].items():
            if sha(Path(item['snapshot'])/name)!=bound:raise ValueError('TEXT_ENCODER_ASSET_CHANGED')
    encoderhash=digest(identity);namespace='g1_text_v1_'+encoderhash[:12];root=ROOT/'data/text'/namespace;root.mkdir(parents=True,exist_ok=True)
    captions={};relations=[]
    for row in records:
        if row['project_split']!='train' or row['status']!='QA_PASS' or row['target_fps']!=25:raise ValueError('TEXT_CACHE_REQUIRES_ACCEPTED_G1_TRAIN')
        spans=row.get('caption_time_spans') or [{'text':text,'native_start_sec':row['native_start_sec'],'native_end_sec':row['native_end_sec'],'whole_motion':True} for text in row.get('captions',[])]
        for span in spans:
            text=span['text']
            if not text.strip():continue
            key=digest({'text':text,'encoder':encoderhash});captions[key]=text
            relations.append({'sample_id':row['sample_id'],'source_family_id':row['source_family_id'],'caption_id':span.get('caption_id'),
                'native_start_sec':span['native_start_sec'],'native_end_sec':span['native_end_sec'],'text':text,'cache_key':key})
    existingpath=root/'index.json';old=json.loads(existingpath.read_text()) if existingpath.exists() else {};cached=old.get('embeddings',{})
    encoder=None;outputs={}
    for key,text in sorted(captions.items()):
        path=root/'embeddings'/(key+'.npy');previous=cached.get(key)
        if resume and previous and path.exists() and sha(path)==previous['sha256']:
            value=np.load(path)
            if value.shape!=(4096,) or not np.isfinite(value).all():raise ValueError('INVALID_CACHED_TEXT_EMBEDDING')
        else:
            if encoder is None:encoder=ExactMotionMamba2TextEncoder(base_mirror=identity['llama']['snapshot'],mntp_adapter=identity['mntp']['snapshot'],
                supervised_adapter=identity['supervised']['snapshot'],device=device)
            tensor,_=encoder([text]);value=tensor[0,0].float().cpu().numpy()
            if value.shape!=(4096,) or not np.isfinite(value).all():raise ValueError('INVALID_TEXT_ENCODER_OUTPUT')
            save_array(path,value.astype(np.float32))
        outputs[key]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'text':text}
        if len(outputs)%100==0:
            write_json(existingpath,{'status':'RUNNING','encoder_sha256':encoderhash,'embeddings':outputs,'manifest_sha256':sha(manifest)})
            print('text embeddings',len(outputs),'/',len(captions),flush=True)
    report={'status':'COMPLETE','encoder_sha256':encoderhash,'encoder_assets':identity,'embeddings':outputs,'caption_relations':relations,
        'manifest_sha256':sha(manifest),'module_sha256':sha(__file__),'unique_texts':len(outputs),'caption_relations_count':len(relations),
        'dtype':'float32','dimension':4096,'missing_caption_policy':'explicit unconditional sample; no synthesized caption or random hash-vector fallback'}
    write_json(existingpath,report);write_json(ROOT/'audit/current_text_cache.json',{'index_path':str(existingpath.relative_to(ROOT)),'sha256':sha(existingpath)})
    return report


def embedding(index,text):
    key=digest({'text':text,'encoder':index['encoder_sha256']});entry=index['embeddings'].get(key)
    if entry is None:raise ValueError('CAPTION_NOT_IN_FROZEN_TEXT_CACHE')
    path=ROOT/entry['path']
    if sha(path)!=entry['sha256']:raise ValueError('TEXT_EMBEDDING_CHANGED')
    return np.load(path)
