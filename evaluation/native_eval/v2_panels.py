import collections
import re
import numpy as np
from .common import *
from .v2_prepare import DOMAINS
from .hard_negatives import MIRROR,swap_lr

def build():
    marker=ROOT/'panels/SEMANTIC_PANELS.json'
    if marker.exists():return read(marker)
    all_records={d:{s:list(rows(ROOT/'data'/f'{d}_{s}.jsonl')) for s in ('development','holdout')} for d in DOMAINS}
    targets={}
    for d in DOMAINS:
        for split,rr in all_records[d].items():
            for r in rr:
                mid=r['sample_id']+'_M' if d=='bones' else 'humanml3d:M'+r['source_motion_id'];targets[mid]=(d,split,r)
    source={}
    # Legal original train and exposed validation metadata only. No original
    # test features or generator artifacts enter model selection.
    for original in ('train','val'):
        for r in rows(RELEASE/'manifests'/f'accepted_{original}.jsonl'):
            if r['sample_id'] in targets:source[r['sample_id']]=r
    pairs=collections.defaultdict(list);exclusions=[];families=collections.defaultdict(set)
    for mid,(d,split,r) in sorted(targets.items(),key=lambda x:identity(['mirror-v2',x[0]])):
        key=d+':'+split
        if len(pairs[key])>=96 or r['source_family_id'] in families[key]:continue
        mirror=source.get(mid)
        if not mirror:continue
        cp=[(c,mc) for c in r['captions'] for mc in mirror.get('captions',[]) if swap_lr(c)==mc.lower() and c.lower()!=mc.lower()]
        if not cp:continue
        path=Path(mirror['feature_path']);path=path if path.is_absolute() else UPSTREAM/path
        a=np.load(r['feature_path']);b=np.load(path)
        if a.shape!=b.shape:continue
        pa=a[:,5:104].reshape(-1,33,3);pb=b[:,5:104].reshape(-1,33,3)
        pa=np.concatenate([np.zeros((len(a),1,3)),pa],axis=1);pb=np.concatenate([np.zeros((len(a),1,3)),pb],axis=1)
        reflected=pa[:,MIRROR].copy();reflected[:,:,0]*=-1;err=float(np.sqrt(np.mean((reflected-pb)**2)))
        if err>.025:exclusions.append(dict(sample_id=r['sample_id'],reason='numeric mirror mismatch',rmse=err));continue
        if sha(path)!=mirror['features_hash']:raise ValueError('Mirror hash changed')
        c,mc=cp[0];mr=dict(r,sample_id=mid,feature_path=str(path),caption=mc,captions=[mc],actual_feature_sha256=sha(path))
        pairs[key].append(dict(original=dict(r,caption=c),mirror=mr,position_rmse=err,evidence='source-provided matched mirror text + reflected/swapped G1 positions <=2.5cm; not arbitrary text editing'));families[key].add(r['source_family_id'])
    overview={}
    for d in DOMAINS:
        for split in ('development','holdout'):
            key=d+':'+split;write(ROOT/'panels'/f'{d}_{split}_direction.json',pairs[key]);candidates=list(rows(ROOT/'panels'/f'{d}_{split}_order_candidates.jsonl'));lookup={r['sample_id']:r for r in all_records[d][split]};review=[];seen=set()
            for c in sorted(candidates,key=lambda c:identity(c['sample_id'])):
                if c['family'] in seen:continue
                seen.add(c['family']);r=lookup[c['sample_id']];a=np.load(r['feature_path']);n=max(2,len(a)//4)
                speed=np.linalg.norm(np.diff(a[:,[0,2]],axis=0)*25,axis=1)
                review.append(dict(c,kinematic_evidence=dict(early_height=float(a[:n,1].mean()),late_height=float(a[-n:,1].mean()),early_horizontal_speed=float(speed[:n].mean()),late_horizontal_speed=float(speed[-n:].mean())),decision='EXCLUDED_FROM_VERIFIED_ORDER',reason='aggregate height/speed and whole-motion caption cannot establish natural counterfactual A/B temporal alignment'))
            jsonl(ROOT/'panels'/f'{d}_{split}_order_review.jsonl',review)
            overview[key]=dict(direction_pairs=len(pairs[key]),order_caption_candidates=len(candidates),order_families=len(seen),independently_verified_order_pairs=0,order_status='INSUFFICIENT_EVIDENCE')
    write(ROOT/'panels/MIRROR_EXCLUSIONS.json',exclusions);write(marker,overview);return overview

def cache_missing_mirrors():
    from .text import TextCache,build_cache
    current=TextCache();extra=set()
    for d in DOMAINS:
        for s in ('development','holdout'):
            for pair in read(ROOT/'panels'/f'{d}_{s}_direction.json'):
                for r in (pair['original'],pair['mirror']):
                    if r['caption'] not in current.lookup:extra.add(r['caption'])
    if extra:build_cache(splits=(),domains=(),extra_texts=sorted(extra))
    return dict(additional_texts=len(extra),motion_inference=False)
