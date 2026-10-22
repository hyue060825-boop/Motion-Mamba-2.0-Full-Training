"""Frozen general T5 encoder, content-addressed masked-mean features."""
from motion_mamba2.paths import artifact_path
import gc
import os
import time
import numpy as np
import torch
from .common import *

def build_cache(splits=('train','development'), domains=('bones','humanml3d'), extra_texts=()):
    cfg=read(CODE/'config.json')
    prior=read(artifact_path('evaluator_asset_manifest_json_c38780c4a2'))
    assets={name:sha(TEXT/name) for name in ['config.json','model.safetensors','spiece.model']}
    if assets!=prior['text_hashes']:raise ValueError('T5 content identity mismatch')
    spec=dict(backbone='google-t5/t5-3b',revision=prior['text_revision'],assets=assets,pooling='attention-mask mean last_hidden_state, includes EOS, no task prefix',
              tokenizer='SentencePieceProcessor original spiece.model + EOS1, pad0; known-answer verified',text_processing='exact admitted stripped source text; no truncation',max_tokens=cfg['text_max_tokens'],dtype='bfloat16 inference; float32 stored')
    texts=sorted({c for d in domains for s in splits for r in rows(ROOT/'data'/f'{d}_{s}.jsonl') for c in r['captions']}|set(extra_texts))
    key=identity(dict(spec=spec,texts=texts));folder=ROOT/'text'/key;folder.mkdir(parents=True,exist_ok=True)
    index={t:i for i,t in enumerate(texts)};done=folder/'COMPLETE.json'
    if done.exists():return folder
    write(folder/'identity.json',dict(key=key,spec=spec,texts=texts,domains=domains,splits=splits))
    from transformers import T5EncoderModel
    from .tokenizer import FrozenTokenizer,tokenizer_checks
    tok=FrozenTokenizer(TEXT/'spiece.model')
    write(ROOT/'TOKENIZER_CHECKS.json',tokenizer_checks(TEXT/'spiece.model'))
    progress=read(folder/'progress.json') if (folder/'progress.json').exists() else dict(next=0)
    with gpu_budget('text:'+key):
        model=T5EncoderModel.from_pretrained(str(TEXT),local_files_only=True,dtype=torch.bfloat16).eval().to('cuda:0')
        model.requires_grad_(False)
        p=folder/'features.npy'
        a=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype='float32',shape=(len(texts),1024))
        start=time.time();start_i=progress['next'];last_save=start
        for i in range(start_i,len(texts),cfg['text_batch_size']):
            batch=texts[i:i+cfg['text_batch_size']]
            encoded=tok(batch,padding=True,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>cfg['text_max_tokens']:raise ValueError('Overlong text; requires new explicit text policy')
            inputs={k:v.to('cuda:0') for k,v in encoded.items()}
            with torch.inference_mode():
                y=model(**inputs).last_hidden_state.float();mask=inputs['attention_mask'][:,:,None]
                pooled=(y*mask).sum(1)/mask.sum(1)
            a[i:i+len(batch)]=pooled.cpu().numpy()
            if i==start_i+cfg['text_batch_size']*9:
                seconds=(time.time()-start)/(i+len(batch)-start_i)
                write(folder/'ETA.json',dict(seconds_per_text=seconds,remaining_seconds=seconds*(len(texts)-i-len(batch)),texts=len(texts),time=time.time()))
                print('TEXT_ETA',read(folder/'ETA.json'),flush=True)
            if time.time()-last_save>60 or i+len(batch)==len(texts):
                a.flush();write(folder/'progress.json',dict(next=i+len(batch),time=time.time()));resource_guard();last_save=time.time()
        write(done,dict(features_sha256=sha(p),identity_sha256=sha(folder/'identity.json'),texts=len(texts)))
        del model;gc.collect();torch.cuda.empty_cache()
    ledger('text_cache_done',key=key,texts=len(texts),splits=splits)
    return folder

class TextCache:
    def __init__(self,folders=None):
        self.lookup={};self.identities=[]
        # Reuse only the content-addressed, tokenizer-verified v1 T5 feature
        # cache. It is immutable input; motion projections are always trained
        # from random initialization in v2.
        default_folders=sorted((ROOT/'text').glob('*'))+sorted((Path(artifact_path('evaluator_native_67895f3cd5'))/'text').glob('*'))
        prior=read(artifact_path('evaluator_asset_manifest_json_c38780c4a2'))
        expected_spec=None
        for folder in (folders or default_folders):
            if not (folder/'COMPLETE.json').exists():continue
            info=read(folder/'identity.json');done=read(folder/'COMPLETE.json')
            spec=info['spec']
            if spec['revision']!=prior['text_revision'] or spec['assets']!=prior['text_hashes'] or 'SentencePieceProcessor original' not in spec['tokenizer'] or spec['pooling']!='attention-mask mean last_hidden_state, includes EOS, no task prefix':raise ValueError('Incompatible text backbone/tokenizer/pooling')
            if expected_spec is not None and spec!=expected_spec:raise ValueError('Cannot mix incompatible text caches')
            expected_spec=spec
            if sha(folder/'identity.json')!=done['identity_sha256'] or sha(folder/'features.npy')!=done['features_sha256']:raise ValueError('Text cache changed')
            a=np.load(folder/'features.npy',mmap_mode='r')
            for i,t in enumerate(info['texts']):self.lookup[t]=(a,i)
            self.identities.append(done)
    def __getitem__(self,text):
        a,i=self.lookup[text];return np.array(a[i],copy=True)
