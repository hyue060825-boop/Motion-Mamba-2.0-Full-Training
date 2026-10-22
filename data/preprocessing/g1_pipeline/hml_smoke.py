"""Gated HumanML3D native SMPL-H -> locked project G1 smoke batch."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .common import ROOT,HML,read_jsonl,write_jsonl,write_json,sha
from .hml_native_runner import retarget_clip
from .robot import export_csv,encode_csv


def _timestamps(row):
    it=row['native_interval']; fps=float(row['original_fps']); n=int(it['native_end_index_exclusive']-it['native_first_index'])
    if n<2: raise ValueError('HML_TOO_SHORT_NATIVE_INTERVAL')
    last=it['start_sec']+(n-1)/fps
    return it['start_sec']+np.arange(int(np.floor((last-it['start_sec'])/.04+1e-9))+1)*.04


def run():
    from .gates import require_pass
    require_pass('01_bones_calibration')
    manifest=ROOT/'manifests/hml_smoke100.jsonl';rows=read_jsonl(manifest)
    out=ROOT/'work/hml_native_smoke_qpos';csvout=ROOT/'work/hml_native_smoke_csv';out.mkdir(exist_ok=True);csvout.mkdir(exist_ok=True)
    results=[]
    for row in rows:
        try:
            it=row['native_interval']; t=_timestamps(row); q,meta=retarget_clip(HML/row['source_paths']['native'],start_frame=it['native_first_index'],end_frame=it['native_end_index_exclusive'],target_timestamps=t,config_path=ROOT/'configs/bones_soma_rootweight_v6.json')
            qpath=out/(row['source_motion_id']+'.npy');np.save(qpath,q)
            cpath=csvout/(row['source_motion_id']+'.csv');export_csv(cpath,q,t,{'sample_id':row['sample_id'],'source':'humanml3d_native_smplh','adapter':'native_smplh_local_rotation_fk','source_hashes':row.get('source_hashes')})
            feat,rep,_=encode_csv(cpath);inv=rep.inverse(feat,is_normalized=False)
            inverse_finite=all(bool(np.isfinite(v.detach().cpu().numpy()).all()) for v in inv.values() if hasattr(v,'detach'))
            finite=bool(np.isfinite(q).all() and np.isfinite(feat.detach().cpu().numpy()).all() and inverse_finite)
            results.append({'sample_id':row['sample_id'],'status':'PASS' if finite else 'REJECTED','frames':len(q),'qpos_shape':list(q.shape),'features_shape':list(feat.shape),'finite':finite,'inverse_finite':inverse_finite,'height_m':meta['actual_human_height_m'],'qpos_path':str(qpath.relative_to(ROOT)),'csv_path':str(cpath.relative_to(ROOT))})
        except Exception as exc: results.append({'sample_id':row['sample_id'],'status':'ERROR','error':repr(exc)})
        if len(results)%10==0: write_jsonl(ROOT/'manifests/hml_smoke100_results.jsonl',results)
    write_jsonl(ROOT/'manifests/hml_smoke100_results.jsonl',results)
    report={'attempted':len(results),'passed':sum(x['status']=='PASS' for x in results),'rejected':sum(x['status']=='REJECTED' for x in results),'errors':sum(x['status']=='ERROR' for x in results),'status':'PASS' if len(results)==100 and all(x['status']=='PASS' for x in results) else 'FAIL','manifest_sha256':sha(manifest),'adapter':'native_smplh_local_rotation_fk','target':'project_g1_v1'}
    write_json(ROOT/'qa/aggregate/hml_smoke100.json',report);return results

if __name__=='__main__': print(run()[-1])
