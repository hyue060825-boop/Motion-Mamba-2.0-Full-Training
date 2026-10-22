"""Static source mapping only; does not run GMR or robot trajectories."""
import json
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha
from g1_pipeline.hiphi_rest_mapping import mapping


def run():
    manifest=ROOT/'manifests/hiphi_smoke100.jsonl';rows=read_jsonl(manifest)
    out=ROOT/'work/hiphi_native_rest_mapping_v1';out.mkdir(exist_ok=True);results=[]
    for row in rows:
        path=ROOT/'work/hiphi_packages'/(row['source_motion_id']+'.bvh')
        result={'sample_id':row['sample_id'],'status':'ERROR'}
        try:
            cfg,audit=mapping(path,row['height_m']);cp=out/(row['source_motion_id']+'.config.json');ap=out/(row['source_motion_id']+'.audit.json')
            write_json(cp,cfg);write_json(ap,audit)
            result.update(status='STATIC_MAPPING_MEASURED',native_path=str(path.relative_to(ROOT)),native_sha256=sha(path),
                config_path=str(cp.relative_to(ROOT)),config_sha256=sha(cp),audit_path=str(ap.relative_to(ROOT)),audit_sha256=sha(ap),
                scale=cfg['scale_strategy']['ratio'],source_height_m=cfg['scale_strategy']['source_height_m'],
                height_discrepancy_above_10_percent=audit['source_provenance']['height_discrepancy_above_10_percent'],
                max_rest_position_residual_m=max(r['rest_position_residual_m'] for r in audit['mapping']),
                max_rest_rotation_residual_deg=max(r['rest_rotation_residual_deg'] for r in audit['mapping']))
        except Exception as exc:result['error']=repr(exc)
        results.append(result);print(row['sample_id'],result['status'],result.get('error',''),flush=True)
    report={'status':'STATIC_SOURCE_MAPPING_ONLY','expected':100,'measured':sum(r['status']=='STATIC_MAPPING_MEASURED' for r in results),
        'manifest_sha256':sha(manifest),'module_sha256':sha(ROOT/'g1_pipeline/hiphi_rest_mapping.py'),
        'shared_mapping_sha256':sha(ROOT/'g1_pipeline/native_rest_mapping.py'),'robot_motion_retargeting_run':False,'results':results}
    write_json(ROOT/'audit/hiphi_native_rest_mapping_v1.json',report);return report


if __name__=='__main__':print(run()['measured'])
