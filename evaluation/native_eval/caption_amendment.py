"""Before HumanML3D training, quarantine ambiguous official time tags."""
import shutil
from .common import *

def amend():
    marker=ROOT/'CAPTION_TIME_AMENDMENT.json'
    if marker.exists():return finish(read(marker))
    if (ROOT/'runs/humanml3d').exists():raise RuntimeError('HumanML3D training has begun: do not change its admission in place')
    audit=read(ROOT/'CAPTION_SOURCE_CHECK.json');bad={(r['sample_id'],r.get('caption')) for r in audit['failures']}
    archive=ROOT/'evidence/caption_time_before';archive.mkdir(parents=True,exist_ok=True)
    changed=[]
    for split in ['train','development','confirmation']:
        p=ROOT/'data'/f'humanml3d_{split}.jsonl';rr=list(rows(p));out=[]
        shutil.copyfile(p,archive/p.name)
        for r in rr:
            old=r['captions'];r['captions']=[c for c in old if (r['sample_id'],c) not in bad]
            if r['captions']!=old:changed.append(dict(sample_id=r['sample_id'],split=split,removed=[c for c in old if c not in r['captions']],remaining=len(r['captions'])))
            if r['captions']:out.append(r)
        jsonl(p,out)
        panel=ROOT/'panels'/f'humanml3d_{split}.json'
        if panel.exists():shutil.move(panel,archive/panel.name)
    for name in ['CAPTION_SOURCE_CHECK.json','DATA_MANIFEST.json','SPLIT_AUDIT.json']:
        shutil.copyfile(ROOT/name,archive/name)
    (ROOT/'CAPTION_SOURCE_CHECK.json').unlink()
    manifest=read(ROOT/'DATA_MANIFEST.json');manifest['files']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data').glob('*.jsonl')}
    manifest['time_tag_amendment']=changed;write(ROOT/'DATA_MANIFEST.json',manifest)
    policy=dict(reason='39 source captions have NaN/zero tags; one has NaN/2s. Upstream whole_motion=True is not sufficient proof of complete-segment semantics. All 40 ambiguous captions excluded conservatively before any HumanML3D evaluator training.',
                changes=changed,configuration_search=False,model_outcomes_used=False,execution_stage='BONES training continues; HumanML3D not yet started')
    write(marker,policy)
    return finish(policy)

def finish(policy):
    marker=ROOT/'CAPTION_TIME_AMENDMENT.json'
    if policy.get('complete'):return policy
    policy['execution_stage']=policy.pop('stage',policy.get('execution_stage'))
    archive=ROOT/'evidence/caption_time_before'
    if (ROOT/'runs/humanml3d').exists():raise RuntimeError('Cannot finish amendment after HumanML3D training begins')
    if any(r['split']=='train' and r['remaining']==0 for r in policy['changes']):
        p=ROOT/'data/humanml3d_stats.npz'
        if p.exists():shutil.move(p,archive/p.name)
        from .audit import prepare
        prepare()
    import collections
    split_audit=read(ROOT/'SPLIT_AUDIT.json');manifest=read(ROOT/'DATA_MANIFEST.json')
    for split in ['train','development','confirmation']:
        rr=list(rows(ROOT/'data'/f'humanml3d_{split}.jsonl'));key='humanml3d:'+split
        summary=split_audit['admitted'][key]
        summary.update(motions=len(rr),families=len({r['source_family_id'] for r in rr}),actors=len({r['actor_id_namespaced'] for r in rr}),captions=len({c for r in rr for c in r['captions']}),frames=sum(r['frames'] for r in rr))
        original_key='humanml3d:'+('val' if split!='train' else 'train')
        summary['coverage_of_original']=len(rr)/split_audit['original_counts'][original_key]
        manifest['admitted'][key]=summary
    split_audit['caption_time_amendment']=policy['changes'];write(ROOT/'SPLIT_AUDIT.json',split_audit)
    manifest['files']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data').glob('*.jsonl')};write(ROOT/'DATA_MANIFEST.json',manifest)
    from .train import panel
    for split in ['development','confirmation']:panel('humanml3d',split)
    from .caption_source_check import check
    check()
    # Recreate only auxiliary hard-negative admission files, not scientific model outcomes.
    hard=ROOT/'hard_negatives'
    if not (archive/'hard_negatives').exists():shutil.move(hard,archive/'hard_negatives')
    from .hard_negatives import prepare_hard_negatives
    prepare_hard_negatives()
    policy['complete']=True;write(marker,policy);ledger('caption_time_admission_amendment',**policy)
    return policy

if __name__=='__main__':amend()
