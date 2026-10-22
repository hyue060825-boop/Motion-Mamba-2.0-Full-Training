"""Read existing final manifests/metrics; no conversion or admission changes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,collections,concurrent.futures
from pathlib import Path
import numpy as np
from g1_pipeline.common import ROOT,write_json,sha,now
from g1_pipeline.kinematic_policy import BASE,load


def measure(row):
    m=load(ROOT/row['metrics_path']);h=m.get('human_target_metrics',{})
    c=load((ROOT/row['qpos_path']).parent/'contract.json')
    values={'root_lift_m':(m.get('ground_correction') or {}).get('lift_max_m',0),
            'hinge_correction_rad':(m.get('rate_projection') or {}).get('correction_max_rad',0),
            'inverse_position_max_m':c['inverse_position_max_m']}
    for name,keys in [('pelvis',['pelvis']),('ankle',['left_ankle_roll_link','right_ankle_roll_link']),('wrist',['left_wrist_yaw_link','right_wrist_yaw_link'])]:
        v=[h[k]['position_p95_m'] for k in keys if k in h]
        if v:values[name+'_position_p95_m']=max(v)
    return row['dataset'],row['sample_id'],values,c['status']=='PASS' and c['finite']


def run():
    counts=collections.defaultdict(collections.Counter);humans=[];bindings={}
    for split in ('train','val','test'):
        p=BASE/'manifests'/f'accepted_{split}.jsonl';bindings[str(p.relative_to(ROOT))]=sha(p)
        with p.open() as f:
            for line in f:
                r=json.loads(line);counts[r['dataset']][split]+=1;counts[r['dataset']]['frames']+=r['target_frames']
                if r['dataset']!='bones':humans.append({k:r[k] for k in ('dataset','sample_id','metrics_path','qpos_path')})
    metrics=collections.defaultdict(lambda:collections.defaultdict(list));extreme={};issues=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for source,sid,values,valid in pool.map(measure,humans,chunksize=50):
            if not valid:issues.append(sid)
            for key,value in values.items():
                metrics[source][key].append(value)
                if (source,key) not in extreme or value>extreme[source,key]['value']:extreme[source,key]={'sample_id':sid,'value':value}
    summary={s:{k:{'clips':len(v),'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95)),'max':float(max(v)),'zero_count':sum(x==0 for x in v)} for k,v in group.items()} for s,group in metrics.items()}
    reasons=collections.defaultdict(collections.Counter)
    with (BASE/'manifests/all_decisions.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['decision'] in ('REJECTED','REVIEW_PENDING','ERROR'):
                for why in r['reasons']:
                    if 'NotWithinConfigurationLimits' in why:why='IK_CONFIGURATION_LIMIT'
                    reasons[r['dataset']+':'+r['decision']][why]+=1
    report={'time':now(),'scope':'all final accepted train/val/test manifest entries; human-source fidelity from existing full-frame metrics; no new solver runs',
            'manifest_counts':{k:dict(v) for k,v in counts.items()},'human_source_metrics':summary,'human_contract_issues':issues,
            'extremes':{s:{k:v for (src,k),v in extreme.items() if src==s} for s in metrics},
            'nonaccepted_reasons':{k:dict(v) for k,v in reasons.items()},'bindings':bindings,
            'metric_definition':'Per-clip full-frame position-error p95 against scaled offset GMR human target body origins, world coordinates, worse side for ankles/wrists. Aggregate p50/p95 across clips, not frame-weighted.',
            'final_semantic_review_complete':False,'training_ready':False}
    write_json(BASE/'quality_full_20260919.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
