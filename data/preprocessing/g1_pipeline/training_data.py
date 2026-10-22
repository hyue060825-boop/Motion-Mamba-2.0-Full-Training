"""G1-only wrapper around the actual project dataset and balanced source sampling."""
import json
from collections import defaultdict
import numpy as np
import torch
from .common import ROOT,read_jsonl,sha,setup_imports
setup_imports()
from motion_data.humanml3d import HumanML3DMotionMamba2Dataset

SOURCES=('bones','humanml3d','hiphi')


class G1Dataset(torch.utils.data.Dataset):
    def __init__(self,manifest,stats_dir,*,frames=104,horizon=52,text_cache=None):
        self.records=read_jsonl(manifest)
        self.text_cache=json.loads(text_cache.read_text()) if text_cache is not None else None
        if self.text_cache is not None and (self.text_cache.get('status')!='COMPLETE' or self.text_cache['manifest_sha256']!=sha(manifest)):
            raise ValueError('INCOMPLETE_OR_STALE_TEXT_CACHE')
        if not self.records:raise ValueError('EMPTY_G1_MANIFEST')
        contract=json.loads((ROOT/'contracts/motion_mamba2_414.json').read_text())
        from pathlib import Path
        stats_dir=Path(stats_dir)
        if sha(stats_dir/'mean.npy')!=contract['mean_hash'] or sha(stats_dir/'std.npy')!=contract['std_hash']:raise ValueError('G1_NORMALIZER_CHECKPOINT_MISMATCH')
        for r in self.records:
            if r.get('dataset') not in SOURCES or r.get('target_fps')!=25 or r.get('status')!='QA_PASS' or r.get('representation')!=414 or r.get('skeleton')!='G1Skeleton34':
                raise ValueError('NON_G1_OR_UNACCEPTED_TRAINING_RECORD:'+str(r.get('sample_id')))
            if r.get('project_split')!='train':raise ValueError('NON_TRAIN_SAMPLE_IN_TRAINING_LOADER')
            if r.get('dataset')=='bones' and r.get('gmr_role')!='official_g1_main_pool':raise ValueError('BONES_CANDIDATE_NOT_MAIN_POOL')
        self.loader=HumanML3DMotionMamba2Dataset(manifest,stats_dir,frames=frames,horizon=horizon,
            control_probability=0,condition_dropout=0,text_dropout=0,history_probability=0)
        if self.loader.mean.shape!=(414,) or self.loader.std.shape!=(414,):raise ValueError('G1_EXPLICIT_STATS_SHAPE')

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        index,seed=index if isinstance(index,tuple) else (index,index)
        meta=self.records[index];batch=self.loader.sample(index,int(seed));motion=batch['motion']
        length=int(batch['length']);mask=torch.arange(len(motion))<length
        if motion.shape[1]!=414 or motion.dtype!=torch.float32 or not torch.isfinite(motion[mask]).all():raise ValueError('INVALID_G1_TRAINING_TENSOR')
        if meta.get('caption_time_spans'):
            first=meta['native_start_sec']+batch['crop_start']/25
            end=first+(length-1)/25
            captions=[r['text'] for r in meta['caption_time_spans'] if r['native_start_sec']<=first+1e-8 and r['native_end_sec']>end-1e-8]
        else:captions=meta.get('captions',[])
        # A caption is used only where its physical interval covers this crop.
        # Missing eligible text is explicit and never replaced by invented text.
        rng=np.random.default_rng(seed);batch['text']=captions[int(rng.integers(len(captions)))] if captions else ''
        batch.update(dataset=meta['dataset'],source_family_id=meta['source_family_id'],valid_mask=mask,
            caption_available=bool(captions),representation=414,fps=25,skeleton='G1Skeleton34')
        if self.text_cache is not None:
            from .text_cache import embedding
            batch['text_feature']=torch.from_numpy(embedding(self.text_cache,batch['text']).copy()).float() if captions else torch.zeros(4096)
            batch['text_dropped']=not bool(captions)
        return batch


class SourceFamilySampler(torch.utils.data.Sampler):
    """Choose source, then family, then one eligible motion; mirrors share a family."""
    def __init__(self,records,*,samples,weights=None,seed=20260915,epoch=0):
        self.groups={s:defaultdict(list) for s in SOURCES}
        for index,row in enumerate(records):self.groups[row['dataset']][row['source_family_id']].append(index)
        if any(not group for group in self.groups.values()):raise ValueError('MIXED_SAMPLER_MISSING_SOURCE')
        weights=weights or {'bones':1/3,'humanml3d':1/3,'hiphi':1/3}
        if set(weights)!=set(SOURCES):raise ValueError('MIXED_SAMPLER_SOURCE_WEIGHTS')
        self.probabilities=np.array([weights[s] for s in SOURCES],dtype=float)
        if not np.isfinite(self.probabilities).all() or np.any(self.probabilities<=0):raise ValueError('INVALID_SOURCE_WEIGHT')
        self.probabilities/=self.probabilities.sum();self.samples=int(samples);self.seed=seed;self.epoch=epoch
        if self.samples<1:raise ValueError('EMPTY_SAMPLER_EPOCH')
        self.families={s:sorted(g) for s,g in self.groups.items()}

    def __len__(self):return self.samples

    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]))
        for _ in range(self.samples):
            source=SOURCES[int(rng.choice(len(SOURCES),p=self.probabilities))]
            families=self.families[source];family=families[int(rng.integers(len(families)))];indices=self.groups[source][family]
            yield (indices[int(rng.integers(len(indices)))],int(rng.integers(2**63-1)))

    def set_epoch(self,epoch):self.epoch=int(epoch)
