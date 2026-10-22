"""Deterministic fresh crops/noise, same family isolation and sampling marginals.

The manifest is compact: it freezes candidate groups and generates records by
update index, rather than storing 12.8 million repeated JSON descriptions.
"""
import copy, pickle, hashlib
from pathlib import Path
import numpy as np
from motion_training import data
from motion_training.schema import VERSION, validate_record
from motion_training.io import read, write, sha
from motion_training._compat.fusion_index import caption_task_tags
from g1_motion_contract import CONTRACT

class Plan:
    def __init__(self, header):
        self.header=header
        with open(header['groups_path'],'rb') as f: self.groups=pickle.load(f)
        self.seed=header['seed']; self.batch=header['batch_size']
    def __len__(self): return self.header['planned_exposures']
    def __getitem__(self, s):
        assert isinstance(s,slice) and s.step is None and s.start%self.batch==0
        assert s.stop-s.start==self.batch and s.stop<=len(self)
        return self.records_at(s.start//self.batch, self.batch)
    def records_at(self, step, count, layout=None):
        block=step//20
        layout_rng=np.random.default_rng(self.seed+1000003*block)
        layouts=[v for v,n in zip(data.BUCKETS,[2,5,4,2,4,3]) for _ in range(n)]
        layout_rng.shuffle(layouts); h,f=layout or layouts[step%20]
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step, h,f]))
        sources=[]; conditions=[]
        while len(sources)<count:
            cycle=['bones']*14+['humanml3d']*3+['hiphi']*3; rng.shuffle(cycle); sources+=cycle
        while len(conditions)<count:
            cycle=['none']*5+[v for v in data.CONDITIONS[1:] for _ in range(4)]; rng.shuffle(cycle); conditions+=cycle
        result=[]; nonempty=0
        for lane,(source,condition) in enumerate(zip(sources[:count],conditions[:count])):
            keys,by=self.groups[source,h,f]; family=keys[int(rng.integers(len(keys)))]
            rr=by[family]; r,opts=rr[int(rng.integers(len(rr)))]; lo,hi,caption,span=opts[int(rng.integers(len(opts)))]; start=int(rng.integers(lo,hi+1))
            future=False
            if f and condition!='none': future=nonempty%2==0; nonempty+=1
            i=step*self.batch+lane
            record=validate_record(dict(schema=VERSION,id=f'{self.header["recipe"]}:{i}',source_family_id=family,
                dataset=source,split='TRAIN',feature_path=r['feature_path'],sample_id=r['sample_id'],
                crop_start=start,crop_end=start+h+52+f,caption=caption,caption_span=span,
                caption_tags=caption_task_tags(caption),native_start_sec=r.get('native_start_sec',0),
                supervision='GT',H=h,F=f,G=52,fps=25,skeleton='g1',representation='physical414',
                normalizer_version=CONTRACT,codec_version='g1_history_codec_104_v1',condition=condition,
                out_of_horizon=future,seed=int(self.seed+1000+i),quality_status=r.get('status'),
                retarget_quality=r.get('kinematic_status'),source_motion_id=r.get('source_motion_id'),
                valid_frames=h+52+f,earlier_prefix_frames=0,horizontal_rotation_rad=0.,
                augmentation='identity; world translation applied jointly to motion and conditions'))
            result.append(record)
        return result

def load(path):
    h=read(path)
    if h.get('format')=='lazy_family_crops_v1': h['plan']=Plan(h)
    return h

def prepare(cfg,dest):
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True); p=dest/'MANIFEST.json'
    template=read(cfg['lazy_manifest_template'])
    if p.exists():
        h=read(p)
        assert h==template and sha(h['groups_path'])==h['groups_sha256']
        assert sha(cfg['data_index'])==h['index_sha256'] and sha(cfg['exclusions'])==h['exclusions_sha256']
        return h
    assert sha(template['groups_path'])==template['groups_sha256']
    write(p,template); return template
