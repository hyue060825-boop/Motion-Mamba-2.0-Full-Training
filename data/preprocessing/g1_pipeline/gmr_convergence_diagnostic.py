"""Measure remaining weighted first-frame residual in the completed dev200 run."""
import json
import numpy as np
from .common import ROOT,read_jsonl,write_json,sha
from .gmr_soma import parse
from .locked_target import create_retargeter,prepare
from .gmr_weighted_convergence import converge


def run():
    config=ROOT/'configs/bones_soma_rootweight_v6.json';cfg=json.loads(config.read_text())
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';xml,_=prepare();rows=[]
    for row in read_jsonl(manifest):
        mid=row['source_motion_id'];bvh=ROOT/'work/bones_pairs_dev'/(mid+'.bvh')
        frames,_=parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=np.array([0.]),rotation_basis='world_only')
        solver=create_retargeter(config,xml)
        original=ROOT/'work/bones_soma_rootweight_dev_v6'/(mid+'.npy');q=np.load(original)[0]
        solver.update_targets(frames[0],offset_to_ground=False);solver.configuration.update(q)
        result=converge(solver,solver.tasks2)
        result.update(sample_id=row['sample_id'],source_sha256=sha(bvh),v6_candidate_sha256=sha(original),
                      root_translation_change_m=float(np.linalg.norm(solver.configuration.q[:3]-q[:3])),
                      hinge_change_max_deg=float(np.rad2deg(np.abs(solver.configuration.q[7:]-q[7:])).max()))
        rows.append(result)
        write_json(ROOT/'audit/gmr_convergence_dev_diagnostic.json',{'status':'DEVELOPMENT_ONLY','manifest_sha256':sha(manifest),
            'config_sha256':sha(config),'code_sha256':sha(ROOT/'g1_pipeline/gmr_weighted_convergence.py'),
            'settings':{'max_iterations':50,'relative_tolerance':1e-6},
            'scope':'first frame of every fixed dev clip; stage 2 continuation from v6 candidate, no official reference used',
            'results':rows})
        print(row['sample_id'],result['initial_cost'],result['final_cost'],result['accepted_iterations'],flush=True)
    return rows

if __name__=='__main__':run()
