"""All seven v19 flat-core ground failures: full-clip development correction."""
import io
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT,BONES,read_jsonl,write_json,sha,atomic_bytes
from g1_pipeline.gates import require_pass
from g1_pipeline.scene_policy import classify
from g1_pipeline.ground_clearance import correct
from g1_pipeline.robot import compile_target,read_csv,FORMAT
from g1_pipeline.resample import qpos as resample
from g1_pipeline.bones_continuity_qa import metrics as continuity


def run():
    require_pass('00_preflight')
    namespace='bones_ground_clearance_v22';out=ROOT/'work'/namespace
    if out.exists():raise ValueError('DIAGNOSTIC_NAMESPACE_EXISTS')
    out.mkdir();base='bones_collision_dev_v19';runpath=ROOT/'qa/aggregate'/f'{base}.json'
    source={r['sample_id']:r for r in json.loads(runpath.read_text())['results']}
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';meta={r['sample_id']:r for r in read_jsonl(manifest)}
    qapath=ROOT/'work'/base/'pair_qa.json';qa=json.loads(qapath.read_text())['results']
    selected=[r['sample_id'] for r in qa if classify(meta[r['sample_id']])['core_eligible'] and
              (r['candidate_geometry']['ground_penetration_p95_m']>.02 or r['candidate_geometry']['ground_penetration_max_m']>.05)]
    if len(selected)!=7:raise ValueError('GROUND_DIAGNOSTIC_PANEL_CHANGED')
    model,asset=compile_target();results=[]
    design={'panel':'dev','sample_ids':selected,'expected_panel':200,'selected':len(selected),
            'selection':'all seven v19 ground-penetration failures in metadata flat-core candidates',
            'transform':'minimal smooth upward root-z lift against compiled floor; original root height difference remains in paired errors',
            'permitted_penetration_m':.015,'source_reference_not_used_in_correction':True,'production_adoption':False,
            'hashes':{str(p):sha(p) for p in [runpath,qapath,manifest,ROOT/'g1_pipeline/ground_clearance.py',ROOT/'g1_pipeline/scene_policy.py',ROOT/'scripts/diagnose_bones_ground_clearance.py']},
            'asset_hash':asset['asset_hash'],'resources':{'workers':1,'gpu':False}}
    write_json(out/'run.json',design)
    for sid in selected:
        row=meta[sid];mid=row['source_motion_id'];bound=source[sid]
        qp=ROOT/'work'/base/(mid+'.npy');tp=qp.with_suffix('.time.json')
        if sha(qp)!=bound['output_sha256'] or sha(tp)!=bound['time_sha256']:raise ValueError('BASE_HASH_CHANGED')
        q=np.load(qp);t=np.asarray(json.loads(tp.read_text())['source']['target_timestamps']);output,info=correct(model,q)
        op=out/(mid+'.npy');buf=io.BytesIO();np.save(buf,output);atomic_bytes(op,buf.getvalue());atomic_bytes(out/(mid+'.time.json'),tp.read_bytes())
        write_json(out/(mid+'.solver.json'),info)
        ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps']);ref,_,_=resample(ref,rt,model.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
        align=bound['fixed_planar_alignment'];turn=R.from_euler('z',align['yaw_rad']);shift=np.array(align['translation'])
        if shift[2]!=0:raise ValueError('ROOT_HEIGHT_ALIGNMENT_FORBIDDEN')
        entry={**bound,'root_rmse_aligned_m':float(np.sqrt(np.mean(np.sum((turn.apply(output[:,:3])+shift-ref[:,:3])**2,axis=1)))),
               'root_rmse_raw_m':float(np.sqrt(np.mean(np.sum((output[:,:3]-ref[:,:3])**2,axis=1)))),
               'output_sha256':sha(op),'solver_sha256':sha(out/(mid+'.solver.json')),
               'candidate_continuity':continuity(model,output,t),'ground_lift_max_m':info['lift_max_m'],'ground_lift_mean_m':info['lift_mean_m']}
        results.append(entry);print(sid,info['lift_max_m'],entry['root_rmse_aligned_m'],flush=True)
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',{'status':'DEVELOPMENT_SUBSET','fixed_panel_size':200,'actual_attempted':len(results),'run_config':design,'results':results})
    from g1_pipeline.bones_pair_qa import run as pair_qa
    from g1_pipeline.summarize_bones_qa import summarize
    pair_qa(namespace,'dev',sample_ids=selected);summarize(namespace)


if __name__=='__main__':run()
