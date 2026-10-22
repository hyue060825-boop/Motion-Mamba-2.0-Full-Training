"""Per-joint paired angle errors; no angle wrapping for bounded hinges."""
import json,numpy as np
from .common import ROOT,BONES,read_jsonl,write_json,sha
from .robot import read_csv,FORMAT,compile_target
from .resample import qpos as resample


def run(namespace='bones_soma_rest_dev_v4'):
    if not namespace.replace('_','').isalnum():raise ValueError('INVALID_DIAGNOSTIC_NAMESPACE')
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';m,asset=compile_target();rows=[];values=[]
    for row in read_jsonl(manifest):
        mid=row['source_motion_id'];root=ROOT/'work'/namespace
        try:
            candidate=np.load(root/(mid+'.npy'));t=np.asarray(json.loads((root/(mid+'.time.json')).read_text())['source']['target_timestamps'])
            ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps']);ref,_,_=resample(ref,rt,m.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
            if candidate.shape!=ref.shape:raise ValueError('PAIR_SHAPE_MISMATCH')
            e=np.rad2deg(np.abs(candidate[:,7:]-ref[:,7:]));values.append(e)
            rows.append({'sample_id':row['sample_id'],'status':'MEASURED','frames':len(e),
              'per_joint_mae_deg':e.mean(axis=0).tolist(),'per_joint_p95_deg':np.percentile(e,95,axis=0).tolist(),
              'per_joint_max_deg':e.max(axis=0).tolist(),'source_category':row.get('category'),'body_position':row.get('body_position')})
        except Exception as exc:rows.append({'sample_id':row['sample_id'],'status':'ERROR','reason':repr(exc)})
    measured=[r for r in rows if r['status']=='MEASURED'];pooled=np.concatenate(values)
    clip_mae=np.asarray([r['per_joint_mae_deg'] for r in measured])
    result={'namespace':namespace,'scope':'all fixed development clips, physical bounded hinge values, no modulo wrap',
      'input_manifest_sha256':sha(manifest),'asset_hash':asset['asset_hash'],'expected':200,'measured':len(measured),
      'frames':len(pooled),'joint_names':[m.joint(i).name for i in range(1,m.njnt)],
      'clip_macro_joint_mae_deg':clip_mae.mean(axis=0).tolist(),'across_clip_joint_mae_p95_deg':np.percentile(clip_mae,95,axis=0).tolist(),
      'frame_weighted_joint_mae_deg':pooled.mean(axis=0).tolist(),'pooled_frame_joint_error_p95_deg':np.percentile(pooled,95,axis=0).tolist(),
      'joint_error_max_deg':pooled.max(axis=0).tolist(),'results':rows}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_joint_errors.json',result)
    print('joint errors',len(measured),len(pooled),flush=True);return result
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--namespace',required=True)
    run(parser.parse_args().namespace)
