"""Human-only arm warm-start experiment on recorded development failures."""
import json
import numpy as np
from .common import ROOT,read_jsonl,write_json,sha
from .gmr_soma import parse
from .locked_target import create_retargeter
from .gmr_arm_seed import arm_seed
from .gmr_weighted_convergence import cost,converge


def run():
    selection=ROOT/'audit/bones_arm_branch_diagnostic.json';prior=json.loads(selection.read_text())
    cfgpath=ROOT/'configs/bones_soma_rootweight_v6.json';cfg=json.loads(cfgpath.read_text())
    meta={r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')};results=[]
    for item in prior['results']:
        row=meta[item['sample_id']];mid=row['source_motion_id'];base=ROOT/'work/bones_soma_rootweight_dev_v6'
        q=np.load(base/(mid+'.npy'));times=np.array(json.loads((base/(mid+'.time.json')).read_text())['source']['target_timestamps'])
        frames,_=parse(ROOT/'work/bones_pairs_dev'/(mid+'.bvh'),{n:n for n in cfg['human_scale_table']},target_timestamps=times,rotation_basis='world_only')
        solver=create_retargeter(cfgpath,ROOT/'assets/project_g1_v1/g1.xml');measures=[]
        # Fixed first/middle/last frame of each selected full development clip.
        for ix in sorted(set([0,len(q)//2,len(q)-1])):
            solver.update_targets(frames[ix]);solver.configuration.update(q[ix]);initial=cost(solver.configuration,solver.tasks2)
            details=[]
            for side in ('left','right'):
                _,info=arm_seed(solver,side);details.append(info)
            continuation=converge(solver,solver.tasks2)
            measures.append({'frame':ix,'initial_cost':initial,'final_cost':cost(solver.configuration,solver.tasks2),
                             'seed_details':details,'gmr_continuation':continuation,
                             'final_qpos':solver.configuration.q.tolist()})
        results.append({'sample_id':row['sample_id'],'measurements':measures})
        print(row['sample_id'],[(m['frame'],round(m['initial_cost']),round(m['final_cost'])) for m in measures],flush=True)
        write_json(ROOT/'audit/bones_arm_seed_diagnostic.json',{'status':'DEVELOPMENT_SINGLE_FRAME_DIAGNOSTIC','selection_sha256':sha(selection),
            'config_sha256':sha(cfgpath),'code_sha256':sha(ROOT/'g1_pipeline/gmr_arm_seed.py'),'results':results,
            'continuity_validated':False,'no_reference_input_to_optimizer':True,'not_used_for_release':True})
    return results

if __name__=='__main__':run()
