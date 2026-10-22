"""Audit fixed BONES dev source posture without reading candidate robot poses."""
import json
import numpy as np
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha
from g1_pipeline.gmr_soma import parse
from g1_pipeline.scene_policy import classify
from g1_pipeline.source_posture import support_evidence


def run():
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';rows=read_jsonl(manifest);results=[]
    names=['Hips','Chest','LeftShin','RightShin','LeftFoot','RightFoot']
    design={'scope':'human-only input geometry; no robot qpos/official reference/QA metrics are read',
            'expected':200,'uniform_soma_height_assumption_m':1.8,
            'thresholds':{'relative_pelvis_height_max':.2,'knee_above_hips_min':.025,'torso_up_cosine_max':.5,'minimum_fraction':.1,'minimum_contiguous_s':.2},
            'interpretation':'review flags only; no existing scene assignments or denominators changed',
            'hashes':{str(p):sha(p) for p in [manifest,ROOT/'g1_pipeline/source_posture.py',ROOT/'g1_pipeline/gmr_soma.py',ROOT/'scripts/audit_bones_source_posture.py']}}
    write_json(ROOT/'audit/bones_source_posture_design_v1.json',design)
    for row in rows:
        path=ROOT/'work/bones_pairs_dev'/(row['source_motion_id']+'.bvh')
        frames,fps=parse(path,{n:n for n in names},rotation_basis='world_only')
        p=np.array([[frame[n][0] for n in names] for frame in frames])
        info=support_evidence(p[:,0],p[:,1],p[:,2:4],p[:,4:6],1.8,dt=1/fps)
        result={'sample_id':row['sample_id'],'source_sha256':sha(path),'metadata_scene':classify(row),
                'source_category':row.get('category'),'source_body_position':row.get('body_position'),'source_fps':fps,**info}
        results.append(result);print(row['sample_id'],info['ground_multisupport_review'],info['flagged_frame_fraction'],flush=True)
    write_json(ROOT/'audit/bones_source_posture_v1.json',{'status':'INPUT_POSTURE_REVIEW_NOT_SCENE_ACCEPTANCE','design':design,'measured':len(results),
        'metadata_core_review_flags':[r['sample_id'] for r in results if r['metadata_scene']['core_eligible'] and r['ground_multisupport_review']],
        'results':results})


if __name__=='__main__':run()
