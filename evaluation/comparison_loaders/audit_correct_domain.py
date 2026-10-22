"""Audit domain-correct saved features, keeping old geometry/timing unchanged."""
from motion_mamba2.paths import artifact_path
import sys,json,csv,time,hashlib
from pathlib import Path
import numpy as np
R=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
sys.path.insert(0,str(R))
from correct_domain_rescore import O,rd,wr,sha,independent_fid,r3

def main():
 binding=rd(O/'ASSET_BINDINGS.json');pools=rd(O/'QUERY_POOLS.json')
 old={(r['model'],r['task']):r for r in csv.DictReader((R/'partial_12_20261021/SUMMARY.csv').open())}
 rows=[];refs={};proof=[];methods=[]
 for method in rd(R/'PLAN.json')['models']:
  for task in ['short52','aligned_w04']:
   d=O/'models'/method/task
   if not (d/'COMPLETE.json').exists():continue
   receipt=rd(d/'COMPLETE.json');assert receipt['assets']==binding
   assert sha(d/'FEATURES.npz')==receipt['features_sha256']
   for g in receipt['geometry']:assert sha(g['path'])==g['sha256']
   with np.load(d/'FEATURES.npz') as z:bank={k:z[k].copy() for k in z.files}
   assert len(bank['sample_ids'])==181
   f=independent_fid(bank['distribution_real'][-1],bank['distribution_generated'][-1])
   assert abs(f-receipt['terminal52_L2_FID'])<1e-9
   real_keys=[k for k in bank if '_real' in k or k.startswith('text_seed')]+['sample_ids','starts']
   if task not in refs:refs[task]={k:bank[k] for k in real_keys}
   refdiff={}
   for k in real_keys:
    if bank[k].dtype.kind in 'US':assert np.array_equal(bank[k],refs[task][k])
    else:
     refdiff[k]=float(np.max(abs(bank[k]-refs[task][k])))
     assert np.allclose(bank[k],refs[task][k],rtol=1e-4,atol=1e-5),(method,task,k,refdiff[k])
   scores={}
   for scope in ['generated_interval','terminal52']:
    for role in ['generated','real']:
     vals=[r3(bank[f'text_seed{s}'],bank[f'{scope}_{role}_seed{s}'],pools) for s in [1701,2903]]
     scores[f'{scope}_{role}_R3_percent']=float(np.mean(vals))
   previous=old.get((method,task))
   if previous is None:
    # MotionStreamer joins only after its original complete suite supplies geometry/timing.
    previous=next(r for r in csv.DictReader((R/'SUMMARY.csv').open()) if r['model']==method and r['task']==task)
   rows.append(dict(model=method,task=task,N=181,CE_cm=float(previous['CE_cm']),FID_L2=f,
    R3_terminal52_diagnostic_percent=scores['terminal52_generated_R3_percent'],
    R3_generated_interval_percent=scores['generated_interval_generated_R3_percent'],
    reference_R3_terminal52_percent=scores['terminal52_real_R3_percent'],
    reference_R3_generated_interval_percent=scores['generated_interval_real_R3_percent'],
    latency_ms_per_window=float(previous['latency_ms_per_window']),SR20_percent=float(previous['SR20_percent']),
    retrieval_scope='source caption; cropped-window semantic qualification not established'))
   proof.append(dict(model=method,task=task,receipt_sha256=sha(d/'COMPLETE.json'),reference_max_deltas=refdiff))
  if any(r['model']==method for r in rows):methods.append(method)
 with (O/'CORRECTED_SUMMARY.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 audit=dict(passed=True,partial=len(methods)!=13,models=methods,models_count=len(methods),N=181,
  correct_domain_assets_verified=True,saved_geometry_hashes_verified=True,
  FID_recomputed_from_saved_L2_features=True,R3_recomputed_from_saved_features=True,
  reference_embeddings_agree_with_tolerance=True,CE_and_latency_preserved=True,
  semantic_qualification='full-source published calibration reproduced; new cohort and cropped windows separately reported; no guaranteed80percent or memory claim',
  assets=binding,proof=proof,summary_sha256=sha(O/'CORRECTED_SUMMARY.csv'),time=time.time())
 wr(O/'CORRECTED_AUDIT.json',audit)
 print(json.dumps({k:audit[k] for k in ['passed','partial','models_count','N']}))
 for row in rows:
  if row['model'] in ['native100','cmdm']:print(json.dumps(row))
if __name__=='__main__':main()
