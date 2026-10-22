"""Source-backed left/right pairs, admitted only after numeric mirror verification."""
import re
import numpy as np
from .common import *
from .schema import canonicalize

MIRROR=[0,8,9,10,11,12,13,14,1,2,3,4,5,6,7,15,16,17,26,27,28,29,30,31,32,33,18,19,20,21,22,23,24,25]

def swap_lr(text):
    return re.sub(r'\b(left|right)\b',lambda m:'right' if m[0]=='left' else 'left',text.lower())

def prepare_hard_negatives():
    dest=ROOT/'hard_negatives';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'MANIFEST.json').exists():return read(dest/'MANIFEST.json')
    source={r['sample_id']:r for r in rows(RELEASE/'manifests/accepted_val.jsonl')}
    result={};excluded=[]
    for d in ['bones','humanml3d']:
      for split in ['development','confirmation']:
        selected=[]
        rr=list(rows(ROOT/'data'/f'{d}_{split}.jsonl'))
        for r in sorted(rr,key=lambda x:identity(['hard-pair',x['sample_id']])):
            mid=r['sample_id']+'_M' if d=='bones' else 'humanml3d:M'+r['source_motion_id']
            mirror=source.get(mid)
            if not mirror:continue
            captions=mirror.get('captions',[])
            pairs=[(c,s) for c in r['captions'] for s in captions if swap_lr(c)==s.lower() and c.lower()!=s.lower()]
            if not pairs:continue
            a=np.load(r['feature_path']);b=np.load(mirror['feature_path'])
            if a.shape!=b.shape:continue
            pa=np.concatenate((a[:,:3,None].transpose(0,2,1),a[:,5:104].reshape(-1,33,3)),axis=1)
            pb=np.concatenate((b[:,:3,None].transpose(0,2,1),b[:,5:104].reshape(-1,33,3)),axis=1)
            pa[:,0,[0,2]]=0;pb[:,0,[0,2]]=0
            reflected=pa[:,MIRROR].copy();reflected[:,:,0]*=-1
            error=float(np.sqrt(np.mean((reflected-pb)**2)))
            if error>.025:
                excluded.append(dict(sample_id=r['sample_id'],reason='mirror_position_rmse_gt_2.5cm',rmse=error));continue
            c,mc=pairs[0]
            mr=dict(r,sample_id=mid,feature_path=mirror['feature_path'],caption=mc,captions=[mc],actual_feature_sha256=sha(mirror['feature_path']))
            if mr['actual_feature_sha256']!=mirror['features_hash']:raise ValueError('Hard-negative mirror hash mismatch')
            selected.append(dict(original=dict(r,caption=c),mirror=mr,mirror_position_rmse=error,category='turn' if 'turn' in c.lower() else 'left_right',validation='same source mirrored counterpart; exact left/right text substitution; physical joint swap and X reflection <=2.5cm RMSE; same duration'))
            if len(selected)>=64:break
        write(dest/f'{d}_{split}.json',selected);result[d+':'+split]=dict(pairs=len(selected),sha256=sha(dest/f'{d}_{split}.json'))
    # Source-text order examples are kept as candidates, not silently treated as verified reversals.
    result['order_status']='PENDING: reversing a complex action is not automatically the semantic opposite; no fabricated action-order ground truth'
    write(dest/'EXCLUSIONS.json',excluded);write(dest/'MANIFEST.json',result);return result

def evaluate_hard(model,domain,split,text,stats):
    from .encoding import encode
    from .metrics import normalize,family_mean_ci
    pairs=read(ROOT/'hard_negatives'/f'{domain}_{split}.json');results=[]
    for pair in pairs:
        records=[pair['original'],pair['mirror']]
        if any(r['caption'] not in text.lookup for r in records):raise ValueError('Missing hard-negative text cache')
        _,m,t=encode(model,records,text,stats)
        sim=normalize(t)@normalize(m).T
        results.append(dict(sample_id=records[0]['sample_id'],family=records[0]['source_family_id'],category=pair['category'],
                            original_margin=float(sim[0,0]-sim[0,1]),mirror_margin=float(sim[1,1]-sim[1,0]),
                            accuracy=(int(sim[0,0]>sim[0,1])+int(sim[1,1]>sim[1,0]))/2))
    margins=[(r['original_margin']+r['mirror_margin'])/2 for r in results]
    return dict(pairs=len(results),results=results,mean_margin_ci=family_mean_ci(margins,[r['family'] for r in results]) if results else None,
                status='VERIFIED_MIRROR_PAIRS' if results else 'NO_VERIFIED_PAIRS',order_status='PENDING_VERIFIED_ORDER_PANEL')
