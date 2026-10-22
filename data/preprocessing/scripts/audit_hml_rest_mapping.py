"""Audit fixed HML100 static rest shapes only; never solve a motion trajectory."""
import json
import numpy as np
from g1_pipeline.common import ROOT,HML,read_jsonl,write_json,sha
from g1_pipeline.native_rest_mapping import smplh_mapping


def run():
    manifest=ROOT/'manifests/hml_smoke100.jsonl';rows=read_jsonl(manifest);out=ROOT/'work/hml_native_rest_mapping_v1';out.mkdir(exist_ok=True)
    results=[]
    for row in rows:
        record={'sample_id':row['sample_id'],'status':'ERROR'}
        try:
            path=HML/row['source_paths']['native'];cfg,audit=smplh_mapping(path,mirror=row['mirror_flag'])
            cp=out/(row['source_motion_id']+'.config.json');ap=out/(row['source_motion_id']+'.audit.json');write_json(cp,cfg);write_json(ap,audit)
            record.update(status='STATIC_MAPPING_MEASURED',config_path=str(cp.relative_to(ROOT)),config_sha256=sha(cp),
                audit_path=str(ap.relative_to(ROOT)),audit_sha256=sha(ap),native_sha256=sha(path),
                source_height_m=cfg['scale_strategy']['source_height_m'],scale=cfg['scale_strategy']['ratio'],
                max_rest_position_residual_m=max(r['rest_position_residual_m'] for r in audit['mapping']),
                max_rest_rotation_residual_deg=max(r['rest_rotation_residual_deg'] for r in audit['mapping']))
        except Exception as exc:record['error']=repr(exc)
        results.append(record);print(record['sample_id'],record['status'],record.get('error',''),flush=True)
    report={'status':'STATIC_SOURCE_MAPPING_ONLY','expected':100,'measured':sum(r['status']=='STATIC_MAPPING_MEASURED' for r in results),
        'robot_motion_retargeting_run':False,'source_model':'native SMPL-H, not relabelled SMPL-X parameters',
        'manifest_sha256':sha(manifest),'module_sha256':sha(ROOT/'g1_pipeline/native_rest_mapping.py'),'results':results}
    write_json(ROOT/'audit/hml_native_rest_mapping_v1.json',report);return report


if __name__=='__main__':print(run()['measured'])
