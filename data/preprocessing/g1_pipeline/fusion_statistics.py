"""Streaming descriptive physical statistics; never replaces model normalization."""
import json
import numpy as np
from .common import ROOT,read_jsonl,write_json,sha
from .gates import require_pass


class Moments:
    def __init__(self,width=414):self.count=0;self.mean=np.zeros(width);self.m2=np.zeros(width)

    def add(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('INVALID_STATS_INPUT')
        if not len(x):return
        other=Moments(x.shape[1]);other.count=len(x);other.mean=x.mean(axis=0);other.m2=((x-other.mean)**2).sum(axis=0);self.merge(other)

    def merge(self,other):
        if not other.count:return
        total=self.count+other.count;delta=other.mean-self.mean
        self.m2+=other.m2+delta**2*self.count*other.count/total;self.mean+=delta*other.count/total;self.count=total

    def result(self):
        if not self.count:raise ValueError('EMPTY_STATS')
        std=np.sqrt(self.m2/self.count)
        return {'frames':self.count,'mean':self.mean.tolist(),'std':std.tolist(),'variance_definition':'population M2/N',
            'near_zero_std_indices':np.flatnonzero(std<1e-6).tolist(),'clamped_for_description':False}


class Reservoir:
    def __init__(self,limit=16384,seed=20260915):self.limit=limit;self.rng=np.random.default_rng(seed);self.keys=np.empty(0);self.values=np.empty((0,414),np.float32);self.seen=0
    def add(self,x):
        self.seen+=len(x);keys=self.rng.random(len(x));combined=np.r_[self.keys,keys];values=np.concatenate([self.values,x])
        keep=np.argpartition(combined,-self.limit)[-self.limit:] if len(combined)>self.limit else np.arange(len(combined))
        self.keys=combined[keep];self.values=values[keep]
    def result(self):
        if not len(self.values):raise ValueError('EMPTY_QUANTILE_SAMPLE')
        return {'sampled_frames':len(self.values),'seen_frames':self.seen,'method':'uniform independent random-priority frame reservoir',
            'quantiles':{str(q):np.quantile(self.values,q,axis=0).tolist() for q in (.01,.05,.5,.95,.99)},
            'approximate':self.seen>len(self.values),'seed':20260915}


def run(manifest=None):
    require_pass('04_fusion_qa');manifest=manifest or ROOT/'manifests/accepted_train.jsonl';rows=read_jsonl(manifest)
    sources=('bones','humanml3d','hiphi');moments={s:Moments() for s in sources};reservoirs={s:Reservoir() for s in sources}
    counts={s:0 for s in sources};motion_mean={s:[] for s in sources};motion_second={s:[] for s in sources};hashes=set();ids=set();excluded=[];bindings={}
    normal=json.loads((ROOT/'contracts/motion_mamba2_414.json').read_text());from pathlib import Path
    chosen=Path(normal['chosen_normalizer']);mean=np.load(chosen/'mean.npy');std=np.load(chosen/'std.npy');idx=np.r_[np.arange(5),np.arange(9,418)]
    if sha(chosen/'mean.npy')!=normal['mean_hash'] or sha(chosen/'std.npy')!=normal['std_hash']:raise ValueError('NORMALIZER_CHANGED')
    mean=mean[idx];std=np.sqrt(std[idx]**2+1e-5);outliers={s:np.zeros(414,dtype=np.int64) for s in sources}
    for row in rows:
        sid=row['sample_id'];source=row['dataset']
        if sid in ids:raise ValueError('DUPLICATE_STATS_SAMPLE_ID')
        ids.add(sid)
        if source not in sources or row['project_split']!='train' or row['status']!='QA_PASS' or row['target_fps']!=25:raise ValueError('STATS_TRAIN_QA_ONLY')
        path=ROOT/row['features_path'];bound=sha(path)
        if bound!=row['features_hash']:raise ValueError('STATS_FEATURE_CHANGED')
        bindings[str(path.relative_to(ROOT))]=bound
        if bound in hashes:excluded.append({'sample_id':sid,'reason':'EXACT_FEATURE_DUPLICATE'});continue
        hashes.add(bound);x=np.load(path,mmap_mode='r',allow_pickle=False)
        if x.shape!=(row['target_frames'],414):raise ValueError('STATS_FEATURE_SHAPE')
        motion=Moments()
        for start in range(0,len(x),8192):
            block=np.asarray(x[start:start+8192]);moments[source].add(block);motion.add(block);reservoirs[source].add(block)
            outliers[source]+=np.sum(abs((block-mean)/std)>5,axis=0)
        counts[source]+=1;motion_mean[source].append(motion.mean);motion_second[source].append(motion.m2/motion.count+motion.mean**2)
    if any(not n for n in counts.values()):raise ValueError('STATS_MISSING_SOURCE')
    merged=Moments()
    for m in moments.values():merged.merge(m)
    equal_mean=np.mean([m.mean for m in moments.values()],axis=0)
    equal_var=np.mean([m.m2/m.count+m.mean**2 for m in moments.values()],axis=0)-equal_mean**2
    mm=np.mean([v for group in motion_mean.values() for v in group],axis=0)
    mv=np.mean([v for group in motion_second.values() for v in group],axis=0)-mm**2
    report={'status':'DESCRIPTIVE_STATS_COMPLETE','representation':414,'fps':25,'train_manifest_sha256':sha(manifest),
        'unique_motion_counts':counts,'exact_duplicate_exclusions':excluded,'source_physical':{s:moments[s].result() for s in sources},
        'merged_frame_weighted':merged.result(),'source_equal_weighted':{'mean':equal_mean.tolist(),'std':np.sqrt(np.maximum(equal_var,0)).tolist()},
        'motion_equal_weighted':{'mean':mm.tolist(),'std':np.sqrt(np.maximum(mv,0)).tolist()},
        'source_frame_weights':{s:moments[s].count/merged.count for s in sources},'training_sampler_source_weights':dict.fromkeys(sources,1/3),
        'source_quantiles':{s:reservoirs[s].result() for s in sources},
        'source_abs_normalized_over5_fraction':{s:(outliers[s]/moments[s].count).tolist() for s in sources},
        'normalizer_replaced':False,'normalizer_hashes':{'mean':normal['mean_hash'],'std':normal['std_hash']},
        'input_feature_hashes':bindings,'module_sha256':sha(__file__)}
    write_json(ROOT/'stats/physical_414.json',report);return report


if __name__=='__main__':print(run()['status'])
