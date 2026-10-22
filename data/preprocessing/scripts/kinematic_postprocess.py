"""Full release accounting, inherited statistics, and bounded runtime verification."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,subprocess,sys,collections
from pathlib import Path
from g1_pipeline.common import ROOT,read_jsonl,write_json,write_jsonl,sha,digest,atomic_bytes,now
from g1_pipeline.kinematic_policy import BASE,load,authorize
from kinematic_finalize import manifests,stats

def selection():
 records=[]
 for split in ['train','val','test']:records.extend(read_jsonl(BASE/'manifests'/f'accepted_{split}.jsonl'))
 strata=collections.defaultdict(list)
 for r in records:
  duration=r['target_frames']/25
  strata[(r['dataset'],r['category'],0 if duration<5 else 1 if duration<30 else 2)].append(r)
 # Fixed seed: one per stratum + per-source duration extremes. No quality-based cherry-picking.
 chosen={}
 for key,rows in sorted(strata.items()):
  r=min(rows,key=lambda r:digest([20260917,'final_semantic_review',r['sample_id']]))
  chosen[r['sample_id']]={'record':r,'reason':'source_category_duration_seeded','stratum':key}
 for source in ['bones','humanml3d','hiphi']:
  group=[r for r in records if r['dataset']==source]
  if group:
   for r in [min(group,key=lambda r:r['target_frames']),max(group,key=lambda r:r['target_frames'])]:chosen.setdefault(r['sample_id'],{'record':r,'reason':'duration_extreme'})
   extremes={}
   for r in group:
    metrics=load(ROOT/r['metrics_path'])
    values={'hinge_correction_rad':(metrics.get('rate_projection') or {}).get('correction_max_rad'),
            'root_lift_m':(metrics.get('ground_correction') or {}).get('lift_max_m'),
            'hinge_speed_rad_s':max(metrics.get('continuity',{}).get('hinge_speed_max_rad_s',[]),default=None),
            'self_collision_m':metrics.get('geometry',{}).get('raw_self_collision_max_depth_m')}
    human=metrics.get('human_target_metrics',{})
    if human:values['human_target_position_p95_m']=max(v['position_p95_m'] for v in human.values())
    for key,value in values.items():
     if value is not None and (key not in extremes or value>extremes[key][0]):extremes[key]=(value,r)
   for key,(value,r) in extremes.items():
    entry=chosen.setdefault(r['sample_id'],{'record':r,'reason':'qa_metric_extreme'})
    entry.setdefault('metric_extremes',{})[key]=value
 path=BASE/'final_review/selection.json';write_json(path,{'seed':20260917,'scope':'stratified accepted clips plus extremes; does not assert exhaustive visual correctness','records':list(chosen.values()),'manifest_bindings':{str((BASE/'manifests'/f'accepted_{s}.jsonl').relative_to(ROOT)):sha(BASE/'manifests'/f'accepted_{s}.jsonl') for s in ['train','val','test']}})
 return path

def run():
 authorize();status=load(BASE/'status.json')
 if status['phase']!='CONVERSION_COMPLETE_FINALIZATION_REQUIRED':raise ValueError('FULL_CONVERSION_NOT_COMPLETE')
 manifests();stats();panel=selection()
 write_json(BASE/'finalization_status.json',{'time':now(),'phase':'MANIFEST_STATS_COMPLETE_RUNTIME_AND_SEMANTIC_REVIEW_PENDING','ready':False,'selection':str(panel.relative_to(ROOT))})
 render_log=(BASE/'final_review/render_process.log').open('ab')
 render=subprocess.run([sys.executable,'scripts/kinematic_render.py'],cwd=ROOT,env=os.environ.copy(),stdout=render_log,stderr=subprocess.STDOUT);render_log.close()
 if render.returncode:raise RuntimeError('FINAL_STRATIFIED_RENDER_FAILED')
 for mode in ['minimal','ardy','hybrid']:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['HF_ENDPOINT']='https://hf-mirror.com'
  log=(BASE/('runtime_'+mode+'.log')).open('ab')
  p=subprocess.run([sys.executable,'scripts/kinematic_runtime.py','--mode',mode],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
  if p.returncode:raise RuntimeError('RUNTIME_SMOKE_FAILED:'+mode)
 write_json(BASE/'finalization_status.json',{'time':now(),'phase':'RUNTIME_COMPLETE_FINAL_SEMANTIC_REVIEW_PENDING','ready':False,'selection':str(panel.relative_to(ROOT))})

if __name__=='__main__':run()
