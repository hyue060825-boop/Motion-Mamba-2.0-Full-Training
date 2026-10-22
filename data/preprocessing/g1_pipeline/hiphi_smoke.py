"""Gated HiPHI Body-only BVH -> locked project G1 smoke batch."""
from __future__ import annotations
import json
from .common import ROOT,read_jsonl,write_json,write_jsonl,sha
from .hiphi_runner import retarget_bvh
from .robot import export_csv,encode_csv
import numpy as np


def run():
    from .gates import require_pass
    require_pass('02_hml100')
    manifest=ROOT/'manifests/hiphi_smoke100.jsonl';rows=read_jsonl(manifest);out=ROOT/'work/hiphi_native_smoke_qpos';csvout=ROOT/'work/hiphi_native_smoke_csv';out.mkdir(exist_ok=True);csvout.mkdir(exist_ok=True);results=[]
    for row in rows:
        try:
            p=ROOT/'work/hiphi_packages'/(row['source_motion_id']+'.bvh')
            q,meta=retarget_bvh(p,config_path=ROOT/'configs/bones_soma_rootweight_v6.json',height_m=row.get('height_m'))
            lines=p.read_text().splitlines();i=lines.index('MOTION');dt=float(lines[i+2].split(':',1)[1]);t=np.arange(len(q))*.04
            qpath=out/(row['source_motion_id']+'.npy');np.save(qpath,q);cpath=csvout/(row['source_motion_id']+'.csv');export_csv(cpath,q,t,{'sample_id':row['sample_id'],'source':'hiphi_body_only','adapter':'hiphi_bvh_channels_local_slerp_fk','source_hash':sha(p)})
            feat,rep,_=encode_csv(cpath);inv=rep.inverse(feat,is_normalized=False);inverse_finite=all(bool(np.isfinite(v.detach().cpu().numpy()).all()) for v in inv.values() if hasattr(v,'detach'));finite=bool(np.isfinite(q).all() and np.isfinite(feat.detach().cpu().numpy()).all() and inverse_finite);results.append({'sample_id':row['sample_id'],'status':'PASS' if finite else 'REJECTED','frames':len(q),'qpos_shape':list(q.shape),'features_shape':list(feat.shape),'finite':finite,'inverse_finite':inverse_finite,'source_fps':meta['source_fps'],'qpos_path':str(qpath.relative_to(ROOT)),'csv_path':str(cpath.relative_to(ROOT))})
        except Exception as exc:results.append({'sample_id':row['sample_id'],'status':'ERROR','error':repr(exc)})
        if len(results)%10==0:write_jsonl(ROOT/'manifests/hiphi_smoke100_results.jsonl',results)
    write_jsonl(ROOT/'manifests/hiphi_smoke100_results.jsonl',results);report={'attempted':len(results),'passed':sum(x['status']=='PASS' for x in results),'rejected':sum(x['status']=='REJECTED' for x in results),'errors':sum(x['status']=='ERROR' for x in results),'status':'PASS' if len(results)==100 and sum(x['status']=='PASS' for x in results)>=95 else 'FAIL','manifest_sha256':sha(manifest),'adapter':'hiphi_bvh_channels_local_slerp_fk','target':'project_g1_v1'};write_json(ROOT/'qa/aggregate/hiphi_smoke100.json',report);return results

if __name__=='__main__':print(run()[-1])
