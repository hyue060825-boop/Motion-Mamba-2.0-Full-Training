"""Materialize a tiny real source-mixed frontend check, separate from final release."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
from g1_pipeline.common import ROOT,BONES,sha,read_jsonl,write_json,write_jsonl
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.kinematic_clip import execute
from kinematic_finalize import record_function
pairs=load(ROOT/'work/bones_core_dev_v24/pair_qa.json')['results']
meta={r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')}
old=load(ROOT/'work/benchmark_equivalence_w1_v1/run.json');output=[];raws=[];record=record_function()
for item in pairs:
 g=item.get('reference_geometry',{})
 if item.get('reference_contract',{}).get('status')!='PASS' or not g:continue
 if g['ground_penetration_p95_m']>.02 or g['ground_penetration_max_m']>.05 or g['raw_self_collision_depth_over_1cm_frame_fraction']>.001 or g['raw_self_collision_max_depth_m']>.03:continue
 row=dict(meta[item['sample_id']]);row.update(sealed_native_path=str(BONES/row['source_paths']['g1']),sealed_native_sha256=sha(BONES/row['source_paths']['g1']))
 out=BASE/'entry_check/bones_frontend'/row['source_motion_id'];out.mkdir(parents=True,exist_ok=True)
 entry,_=execute({'source':'bones','row':row},out,old['asset'],old['code'])
 if entry.get('decision')=='ACCEPTED_KINEMATIC' and entry['frames']>=104:raws.append(entry);break
assert len(raws)==1,'NO_NUMERICALLY_ELIGIBLE_OFFICIAL_BONES_PREFLIGHT'
for short in ['hml','hiphi']:
 ns=load(ROOT/f'audit/current_pilot_{short}100_run.json')['namespace']
 r=next(r for r in read_jsonl(ROOT/'work'/ns/'accepted.jsonl') if r['frames']>=104);raws.append(r)
for r in raws:
 row=dict(r['source_metadata']);row.setdefault('native_start_sec',0);row.setdefault('native_end_sec',r['frames']/25)
 prepared=dict(r,status='KINEMATIC_CANDIDATE',body_only_validated=r['dataset']=='hiphi')
 out=record(row,prepared,pipeline_version='v25_kinematic_training_frontend_preflight',sim_coverage={'status':'NOT_RUN','controller_hash':None},leakage={'status':'PASS','role':'non-final runtime preflight; all original source train'})
 output.append(out)
path=BASE/'runtime_preflight/manifest.jsonl';write_jsonl(path,output);write_json(BASE/'runtime_preflight/source_results.json',{'scope':'non-final frontend check, not a release manifest','results':raws})
print({'status':'PREFLIGHT_INPUTS_READY','sources':[r['dataset'] for r in output],'path':str(path)})
