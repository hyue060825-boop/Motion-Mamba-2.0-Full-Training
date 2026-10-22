"""Apply sealed official dev baselines; never refit on an acceptance panel."""
import json
import numpy as np
from .common import ROOT,BONES,read_jsonl,write_json,sha
from .bones_scene_evidence import load
from .bones_relative_motion_v2 import labels,series
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample


def run(namespace,panel):
    from .calibration_protocol import require_frozen
    policy=require_frozen();path=ROOT/policy['relative_baseline'];baseline=json.loads(path.read_text())['baseline']
    manifest=ROOT/f'manifests/bones_pairs_{panel}.jsonl';scenes={r['sample_id']:r for r in load(manifest)}
    model,_=compile_target();results=[]
    for row in read_jsonl(manifest):
        sid=row['sample_id'];mid=row['source_motion_id'];qp=ROOT/'work'/namespace/(mid+'.npy');tp=qp.with_suffix('.time.json')
        item={'sample_id':sid,'status':'ERROR'}
        try:
            ls=labels({**row,'source_description':scenes[sid]['technical_description']})
            supported=[k for k in ls if baseline[k]['supported']]
            t=np.array(json.loads(tp.read_text())['source']['target_timestamps']);q=np.load(qp)
            rq,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
            rq,_,_=resample(rq,rt,model.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
            roles={}
            for role,pose in [('candidate',q),('reference',rq)]:
                values=series(model,pose,t);roles[role]={}
                for label in supported:
                    roles[role][label]={}
                    for key,a in values.items():
                        threshold=np.asarray(baseline[label]['thresholds'][key]);exceed=a>threshold
                        roles[role][label][key]={'samples':len(a),'any_coordinate_exceed_fraction':float(exceed.any(axis=1).mean()),
                            'per_coordinate_exceed_fraction':exceed.mean(axis=0).tolist(),'per_coordinate_max':a.max(axis=0).tolist(),
                            'max_threshold_ratio':float(np.max(a/threshold)),
                            'flagged_sample_indices':np.flatnonzero(exceed.any(axis=1)).tolist()}
            item.update(status='MEASURED' if supported else 'UNSUPPORTED_CATEGORY',labels=ls,roles=roles,
                        candidate_sha256=sha(qp),time_sha256=sha(tp))
        except Exception as exc:item['error']=repr(exc)
        results.append(item)
    report={'status':'FROZEN_RELATIVE_MEASUREMENTS','baseline_sha256':sha(path),'manifest_sha256':sha(manifest),
            'expected':len(results),'results':results,'no_held_out_threshold_fitting':True}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_relative_motion.json',report);return report
