"""Bounded cross-source physical-neighbour evidence, never universal no-leakage claims."""
import numpy as np
from scipy.spatial import cKDTree
from .common import *

def audit():
    output=ROOT/'CROSS_SOURCE_NEAR_DUPLICATE_SAMPLE.json'
    if output.exists():return read(output)
    candidates=[]
    for r in rows(INDEX):
        if r['dataset']=='hiphi' and not r['mirror_flag']:
            candidates.append(r)
    candidates=sorted(candidates,key=lambda r:identity(['cross-source-review',r['sample_id']]))[:128]
    def feature(r):
        a=np.load(r['feature_path'],mmap_mode='r');positions=a[:,5:104]
        return positions[np.linspace(0,len(a)-1,8).astype(int)].reshape(-1)
    hiphi=np.stack([feature(r) for r in candidates]);tree=cKDTree(hiphi);result={}
    for domain in ['bones','humanml3d']:
        rr=sorted(rows(ROOT/'data'/f'{domain}_development.jsonl'),key=lambda r:identity(['cross-source-review',r['sample_id']]))[:128]
        features=np.stack([feature(r) for r in rr]);distance,ix=tree.query(features,workers=1);closest=np.argsort(distance)[:8]
        result[domain]=[dict(validation_sample_id=rr[i]['sample_id'],generator_train_hiphi_sample_id=candidates[ix[i]]['sample_id'],
           position_rmse_m=float(distance[i]/np.sqrt(features.shape[1])),
           captions=[rr[i]['captions'][0],candidates[ix[i]]['captions'][0]],
           source_families=[rr[i]['source_family_id'],candidates[ix[i]]['source_family_id']],
           actual_hiphi_feature_sha256=sha(candidates[ix[i]]['feature_path']),
           interpretation='physical similarity candidate; source IDs differ, no unproven same-capture conclusion') for i in closest]
    out=dict(hiphi_generator_train_sample_N=len(hiphi),evaluator_validation_N_per_domain=128,results=result,
       limitation='bounded physical position screen; different retargeting/body conventions can obscure copies; unknown cross-source semantic duplicates remain unknown')
    write(output,out);return out

if __name__=='__main__':audit()
