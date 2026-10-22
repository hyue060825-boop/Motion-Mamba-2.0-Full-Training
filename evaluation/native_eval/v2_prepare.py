"""Identity-only admission, immutable text reuse and fresh train statistics."""
from motion_mamba2.paths import artifact_path
import collections
import re
import shutil
import time
import numpy as np
from .common import *
from .schema import canonicalize
from .metrics import candidate_pools

V1=Path(artifact_path('evaluator_native_67895f3cd5'))
DOMAINS=('humanml3d','bones')

def initial_split():
    """Rebuild the exact pre-registered identity split from the immutable v1 input."""
    plan_path=ROOT/'SPLIT_PLAN_V2.json'
    if plan_path.exists() and all((ROOT/'data'/f'{d}_{s}.jsonl').exists() for d in DOMAINS for s in ('train','holdout')):return
    if (ROOT/'CONFIRMATION_OPENED.json').exists():raise ValueError('Cannot silently reconstruct a consumed holdout; restore its frozen manifest')
    cfg=read(CODE/'config.json');plan=dict(version='native-v2',rule='fixed SHA256 identity hash; human family .18, BONES actor .12',generator_unseen_claim=False,domains={})
    for d in DOMAINS:
        rr=list(rows(V1/'data'/f'{d}_train.jsonl'))
        def held(r):
            key=['v2-actor-holdout',r['actor_id_namespaced']] if d=='bones' else ['v2-family-holdout',r['source_family_id']]
            limit=cfg['holdout_fraction_bones_actors'] if d=='bones' else cfg['holdout_fraction_humanml3d']
            return int(identity(key)[:8],16)/2**32<limit
        tr=sorted([dict(r,evaluator_split='v2_train') for r in rr if not held(r)],key=lambda r:r['sample_id'])
        ho=sorted([dict(r,evaluator_split='evaluator_v2_holdout_from_train') for r in rr if held(r)],key=lambda r:r['sample_id'])
        jsonl(ROOT/'data'/f'{d}_train.jsonl',tr);jsonl(ROOT/'data'/f'{d}_holdout.jsonl',ho)
        plan['domains'][d]=dict(train_families=len({r['source_family_id'] for r in tr}),holdout_families=len({r['source_family_id'] for r in ho}),source_sha256=sha(V1/'data'/f'{d}_train.jsonl'))
    write(plan_path,plan)

def one_family(rr):
    chosen={}
    for r in sorted(rr,key=lambda r:identity(['v2-panel',r['sample_id']])):
        chosen.setdefault(r['source_family_id'],dict(r,caption=r['captions'][0]))
    return sorted(chosen.values(),key=lambda r:r['sample_id'])

def make_panel(rr):
    cfg=read(CODE/'config.json');chosen=[];captions=set()
    for r in one_family(rr):
        options=[c for c in r['captions'] if c not in captions]
        if options:
            chosen.append(dict(r,caption=options[0]));captions.add(options[0])
        if len(chosen)>=1024:break
    pools={str(s):candidate_pools(chosen,s) for s in cfg['panel_seeds']}
    matched=[];eligible=[];rng=np.random.default_rng(4101)
    for i,r in enumerate(chosen):
        ids=[j for j,s in enumerate(chosen) if j!=i and s['source_family_id']!=r['source_family_id'] and s['caption']!=r['caption'] and abs(s['frames']-r['frames'])<=max(8,round(r['frames']*.25))]
        if len(ids)>=31:
            eligible.append(i);matched.append(sorted(rng.choice(ids,31,replace=False).tolist()+[i],key=lambda j:chosen[j]['sample_id']))
    p=dict(records=chosen,candidate_pools=pools,length_matched=dict(eligible=eligible,pools=matched),version='native-v2')
    p['panel_sha256']=identity(p);return p

def statistics(domain,rr):
    counts=collections.Counter(r['source_family_id'] for r in rr)
    total=np.zeros(414);sq=total.copy();weight=0.
    for r in rr:
        a=canonicalize(np.load(r['feature_path'])).astype(np.float64);w=1/counts[r['source_family_id']]
        total+=a.mean(0)*w;sq+=(a*a).mean(0)*w;weight+=w
    mean=total/weight;std=np.sqrt(np.maximum(0,sq/weight-mean*mean)).clip(.05)
    path=ROOT/'data'/f'{domain}_stats.npz';np.savez(path,mean=mean.astype('float32'),std=std.astype('float32'))
    write(path.with_suffix('.json'),dict(train_only=True,family_balanced=True,families=len(counts),manifest_sha256=sha(ROOT/'data'/f'{domain}_train.jsonl'),sha256=sha(path),std_floor=.05))

def baseline():
    files={str(p):sha(p) for p in V1.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files.update({str(p):sha(p) for p in Path(artifact_path('evaluator_native_5eb8b43389')).rglob('*') if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts})
    write(ROOT/'V1_READONLY_BASELINE.json',files)
    summaries={}
    for d in DOMAINS:
        summaries[d]={}
        for s in (1701,2903):
            curves=list(rows(V1/'runs'/d/f'seed{s}'/'curve.jsonl'))
            summaries[d][str(s)]={split:read(V1/'validation'/d/f'seed{s}'/split/'RESULT.json') for split in ('development','confirmation')}
            summaries[d][str(s)]['curve_summary']=dict(epochs=len(curves),first=curves[0],last=curves[-1],curve_sha256=sha(V1/'runs'/d/f'seed{s}'/'curve.jsonl'))
    write(ROOT/'BASELINE_REPLAY.json',summaries)

def prepare():
    marker=ROOT/'PREPARE_COMPLETE.json'
    if marker.exists():return read(marker)
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'GPU_BUDGET.json').exists():write(ROOT/'GPU_BUDGET.json',dict(seconds=0.,stages=[],active_since=None,active_stage=None))
    baseline();initial_split();manifest=read(V1/'DATA_MANIFEST.json')
    for name,h in manifest['files'].items():
        if not 'sealed' in name and sha(V1/name)!=h:raise ValueError('v1 manifest changed: '+name)
    original_plan=read(ROOT/'SPLIT_PLAN_V2.json')
    write(ROOT/'SPLIT_PLAN_ORIGINAL_IDENTITY_RULE.json',original_plan)
    exposure=[];audit={};manifests={};fingerprints={};panels={}
    for d in DOMAINS:
        tr=list(rows(ROOT/'data'/f'{d}_train.jsonl'));ho=list(rows(ROOT/'data'/f'{d}_holdout.jsonl'))
        dev=[]
        for split in ('development','confirmation'):
            for r in rows(V1/'data'/f'{d}_{split}.jsonl'):dev.append(dict(r,v1_split=split,evaluator_split='development_v1_exposed'))
        # Same family/source/exact data identity stays in one split. A collision
        # excludes the train family; holdout identities never depend on scores.
        hashes={s:set() for s in ('train','holdout','development')};canon={s:set() for s in hashes}
        for s,rr in [('train',tr),('holdout',ho),('development',dev)]:
            for r in rr:
                h=sha(r['feature_path'])
                if h!=r['actual_feature_sha256']:raise ValueError('Actual motion hash changed '+r['sample_id'])
                a=np.load(r['feature_path'],allow_pickle=False)
                if a.shape!=(r['frames'],414) or not np.isfinite(a).all():raise ValueError('Bad motion')
                signature=__import__('hashlib').sha256(np.round(canonicalize(a),5).tobytes()).hexdigest()
                r['canonical_rounded_sha256']=signature
                hashes[s].add(h);canon[s].add(signature)
                fingerprints[r['sample_id']]=dict(sha256=h,canonical_rounded_sha256=signature)
        holdkeys={r['source_family_id'] for r in ho};devkeys={r['source_family_id'] for r in dev}
        foreign_hashes=hashes['holdout']|hashes['development'];foreign_canon=canon['holdout']|canon['development']
        foreign_sources={h for r in ho+dev for h in (r.get('source_hashes') or {}).values() if h}
        exclude={r['source_family_id'] for r in tr if r['source_family_id'] in holdkeys|devkeys or r['actual_feature_sha256'] in foreign_hashes or r['canonical_rounded_sha256'] in foreign_canon or any(h in foreign_sources for h in (r.get('source_hashes') or {}).values())}
        tr=[r for r in tr if r['source_family_id'] not in exclude]
        for s,rr in [('train',tr),('holdout',ho),('development',dev)]:
            for r in rr:
                exposure.append(dict(sample_id=r['sample_id'],family=r['source_family_id'],actor=r['actor_id_namespaced'],v1_use=r.get('v1_split','train'),v2_use=s,generator_denoiser_family_exposure=r.get('generator_denoiser_family_exposure'),generator_codec_normalizer_exposure=r.get('generator_codec_normalizer_exposure','unknown'),historical_tuning='not exhaustively audited'))
            jsonl(ROOT/'data'/f'{d}_{s}.jsonl',rr)
            p=make_panel(rr);write(ROOT/'panels'/f'{d}_{s}.json',p)
            # Distribution includes all distinct families; repeated/ambiguous
            # captions cannot silently shrink the FID panel.
            dist=one_family(rr);dp=dict(records=dist,role=s,window='center52',version='v2');dp['panel_sha256']=identity(dp);write(ROOT/'panels'/f'{d}_{s}_distribution.json',dp)
            manifests[d+':'+s]=dict(rows=len(rr),families=len({r['source_family_id'] for r in rr}),actors=len({r['actor_id_namespaced'] for r in rr}),sha256=sha(ROOT/'data'/f'{d}_{s}.jsonl'))
            order=[]
            for r in rr:
                for c in r['captions']:
                    triggers=re.findall(r'\b(?:then|before|after|followed by|subsequently|finally)\b',c.lower())
                    if triggers:
                        order.append(dict(sample_id=r['sample_id'],family=r['source_family_id'],actor=r['actor_id_namespaced'],caption=c,caption_source=r['caption_source'],triggers=triggers,directions=re.findall(r'\b(?:left|right|clockwise|counterclockwise)\b',c.lower()),events=re.findall(r'\b(?:walk\w*|run\w*|sit\w*|stand\w*|turn\w*|jump\w*|bend\w*|rais\w*|lift\w*|lower\w*)\b',c.lower()),verification='SOURCE_TEXT_ONLY',time_alignment='whole semantic segment, no verified within-segment event times',exclusion_reason='not independently verified as A->B versus B->A'))
            jsonl(ROOT/'panels'/f'{d}_{s}_order_candidates.jsonl',order)
            panels[d+':'+s]=dict(captions=len(order),families=len({r['family'] for r in order}),status='INSUFFICIENT_EVIDENCE')
        statistics(d,tr)
        audit[d]=dict(train_families=len({r['source_family_id'] for r in tr}),holdout_families=len(holdkeys),excluded_train_families=sorted(exclude),family_overlap=[],actual_feature_overlap=sorted({r['actual_feature_sha256'] for r in tr}&hashes['holdout']),canonical_duplicate_overlap=sorted({r['canonical_rounded_sha256'] for r in tr}&canon['holdout']),actor_overlap=sorted({r['actor_id_namespaced'] for r in tr}&{r['actor_id_namespaced'] for r in ho}),grouping='same source family includes mirrors and all neighboring/overlapping source crops; additional source/exact/canonical hash closure',unknown_near_duplicates='not exhaustively excluded; canonical exact detection is not semantic-near-duplicate certification')
    jsonl(ROOT/'EXPOSURE_LEDGER.jsonl',exposure);write(ROOT/'SPLIT_AUDIT_V2.json',audit);write(ROOT/'DATA_MANIFEST_V2.json',dict(version='native-v2',admitted=manifests,actual_feature_identities=fingerprints,scope='evaluator-only holdout from original train; generator seen/unknown'))
    write(ROOT/'ORDER_PANEL_COUNTS.json',panels)
    for name in ('SKELETON.json','EVAL_CONTRACT.json','TOKENIZER_CORRECTION.json','CAPTION_TIME_AMENDMENT.json','ANNOTATION_AMENDMENT.json'):
        shutil.copyfile(V1/name,ROOT/name)
    from .text import TextCache
    text=TextCache();missing={c for d in DOMAINS for split in ('train','development','holdout') for r in rows(ROOT/'data'/f'{d}_{split}.jsonl') for c in r['captions'] if c not in text.lookup}
    if missing:raise ValueError('Missing frozen text cache')
    write(ROOT/'TEXT_REUSE_AUDIT.json',dict(cache_identities=text.identities,missing=0,projection_weights_reused=False))
    write(marker,dict(completed_at=time.time(),split_audit=audit,text_identity=identity(text.identities)))
    ledger('corrected_setup',reason='initial development panel wrongly used training records; all initial checkpoints excluded',holdout_model_inference_before_correction=False)
    return read(marker)
