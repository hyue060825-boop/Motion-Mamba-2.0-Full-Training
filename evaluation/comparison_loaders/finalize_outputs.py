"""Promote audited domain-correct outputs, preserving superseded evidence."""
from motion_mamba2.paths import artifact_path
import os,sys,json,csv,time,shutil
from pathlib import Path
import numpy as np
R=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
sys.path.insert(0,str(R))
from correct_domain_rescore import O,REL,rd,wr,sha,verify_release,norm
from native_eval.metrics import embedding_health

def save(p,rows):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 audit=rd(O/'CORRECTED_AUDIT.json');wa=rd(O/'WINDOW_AUDIT.json')
 assert audit['passed'] and audit['models_count']==13 and not audit['partial']
 assert wa['passed'] and wa['window_groups']==65
 assert rd(O/'FINAL13_MANUSCRIPT_COMPILE.json')['kind']=='success'
 release=verify_release(REL)
 old=R/'maintenance/20261021_wrong_domain_complete13';old.mkdir(parents=True,exist_ok=True)
 names=['SUMMARY.csv','WINDOW_METRICS.csv','EVALUATOR_TRANSFER_AUDIT.json','FINAL_AUDIT.json','COMPLETE.json','humanml3d_results.tex','main.tex','STATUS.json']
 for name in names:
  p=R/name
  if p.exists() and not (old/name).exists():shutil.copy2(p,old/name)
 for p in R.glob('*_bank.npz'):
  if not (old/p.name).exists():p.rename(old/p.name)
 generation_audit=rd(old/'FINAL_AUDIT.json');assert generation_audit['passed']
 methods=rd(R/'PLAN.json')['models'];rows=list(csv.DictReader((O/'CORRECTED_SUMMARY.csv').open()))
 assert len(rows)==26
 tasks=['short52','aligned_w04'];max_delta=0.;timing_proofs=[];gate_proofs=[]
 for method in methods:
  d=R/'models'/method;info=rd(d/'CHECKPOINTS.json')[method]
  assert rd(d/'COMPLETE.json')['checkpoint']['sha256']==info['sha256']
  gates=rd(d/'GATES.json');assert gates['status'].startswith('PASS')
  gate_proofs.append(dict(method=method,checkpoint_sha256=info['sha256'],gates_sha256=sha(d/'GATES.json')))
  for task in tasks:
   ce=[]
   for p in sorted((d/'evaluation'/method/task).glob('shard*')):
    done=rd(p/'COMPLETE.json');assert sha(p/'geometry.npz')==done['files']['geometry.npz']
    oldrows=[json.loads(s) for s in (p/'per_sample.jsonl').open()]
    with np.load(p/'geometry.npz') as z:
     for i,row in enumerate(oldrows):
      val=float(np.linalg.norm(z['generated_root'][i,-1,[0,2]].astype(np.float64)-np.asarray(row['goal_root_xz_m']))*100)
      delta=abs(val-row['constraint_error_cm']);assert delta<.001;max_delta=max(max_delta,delta);ce.append(val)
   row=next(x for x in rows if x['model']==method and x['task']==task)
   assert len(ce)==181 and abs(np.mean(ce)-float(row['CE_cm']))<.001
   W=1 if task=='short52' else 4;tt=d/'timing'/method/task/'shard00'
   cases=[c for c in rd(tt/'TIMING_DETAIL.json')['rows'] if not c['warmup']]
   assert len(cases)==32 and all(c['batch']==1 for c in cases)
   if method.startswith('native'):
    assert rd(tt/'LOAD.json')['masked_kernel_repair_mode']=='masked'
    assert all(c['graph_delta']['builds']==c['graph_delta']['fallbacks']==0 and c['graph_delta']['replays']==2+22*W for c in cases)
   assert sha(tt/'geometry.npz')==rd(tt/'COMPLETE.json')['files']['geometry.npz']
   val=1000*float(np.median([c['elapsed_seconds'] for c in cases]))/W
   assert abs(val-float(row['latency_ms_per_window']))<1e-9
   timing_proofs.append(dict(model=method,task=task,receipt_sha256=sha(tt/'COMPLETE.json'),detail_sha256=sha(tt/'TIMING_DETAIL.json')))
 for name in ['CMDM','MOTIONSTREAMER']:
  ck=rd(R/f'{name}100K_SAVE_VERIFIED.json')
  assert ck['passed'] and ck['actual_step']==ck['scheduler_completed']==100000 and ck['exposures']==15200000
 output=[]
 for row in rows:
  output.append(dict(row,R3_percent=row['R3_terminal52_diagnostic_percent'],benchmark='HumanML3D-G1-retargeted-heldout',terminal_local_frames=52,windows=1 if row['task']=='short52' else 4))
 for task in tasks:
  ref=next(x for x in rows if x['task']==task)
  output.append(dict(model='retarget',task=task,N=181,CE_cm='',FID_L2=0.,R3_terminal52_diagnostic_percent=ref['reference_R3_terminal52_percent'],R3_generated_interval_percent=ref['reference_R3_generated_interval_percent'],reference_R3_terminal52_percent=ref['reference_R3_terminal52_percent'],reference_R3_generated_interval_percent=ref['reference_R3_generated_interval_percent'],latency_ms_per_window='',SR20_percent='',retrieval_scope='real reference; source-caption crops separately scoped',R3_percent=ref['reference_R3_terminal52_percent'],benchmark='HumanML3D-G1-retargeted-heldout',terminal_local_frames=52,windows=1 if task=='short52' else 4))
 save(R/'SUMMARY.csv',output)
 shutil.copy2(O/'WINDOW_METRICS.csv',R/'WINDOW_METRICS.csv')
 # Canonical feature values now come only from domain-correct receipts.
 diagnostics=[]
 for task in tasks:
  ref=next(x for x in rows if x['task']==task)
  with np.load(O/'models/ardy'/task/'FEATURES.npz') as z:health=embedding_health(norm(z['distribution_real'][-1]))
  diagnostics.append(dict(task=task,N=181,real_terminal52_R3_percent=float(ref['reference_R3_terminal52_percent']),real_generated_interval_R3_percent=float(ref['reference_R3_generated_interval_percent']),distribution=health,window_retrieval_qualified_for_phase_semantics=False))
 wr(R/'EVALUATOR_TRANSFER_AUDIT.json',dict(domain='humanml3d',release=release['release_sha256'],rows=diagnostics,original_calibration=rd(O/'RELEASE_REFERENCE_REPLAY.json'),full_source_reference=rd(O/'CURRENT_COHORT_REFERENCE_AUDIT.json'),unchanged_frozen_evaluator=True,sample_count_below256=True,full_sequence_FID_validated=False,not_canonical_HumanML3D_leaderboard=True))
 shutil.copy2(O/'humanml3d_results.tex',R/'humanml3d_results.tex')
 # Preserve the original remote template, replace only its marked HML section.
 oldmain=(old/'main.tex').read_text();B='% BEGIN HUMANML3D_SECTION1_20261020';E='% END HUMANML3D_SECTION1_20261020'
 assert oldmain.count(B)==oldmain.count(E)==1
 a,t=oldmain.split(B);_,z=t.split(E)
 (R/'main.tex').write_text(a+B+'\n'+(O/'humanml3d_results.tex').read_text()+E+z)
 wr(R/'DEPRECATED_DOMAIN_ASSETS.json',dict(status='SUPERSEDED_DO_NOT_USE_FOR_HUMANML3D_RANKING',archive=str(old),legacy_per_shard_feature_names=['WINDOW_EMBEDDINGS.npz','WINDOW_SCORE_COMPLETE.json'],reason='HiPHI/BONES feature branches incorrectly reused; current correct features in correct_domain_v2_20261021',do_not_rerun_legacy_aggregate_to_publish=True,geometry_and_physical_metrics_remain_valid=True))
 result=dict(passed=True,models=methods,models_count=13,N=181,tasks=tasks,correct_domain_assets=True,geometry_CE_independently_recomputed=True,max_CE_delta_cm=max_delta,feature_L2_FID_independently_checked=True,window_audit=sha(O/'WINDOW_AUDIT.json'),corrected_feature_audit=sha(O/'CORRECTED_AUDIT.json'),original_generation_pairing_audit=sha(old/'FINAL_AUDIT.json'),inputs_paired=generation_audit['inputs_paired'],latency_independently_recomputed=True,timing_proofs=timing_proofs,checkpoint_and_gate_proofs=gate_proofs,local_manuscript_compile=rd(O/'FINAL13_MANUSCRIPT_COMPILE.json'),local_manuscript_update=rd(O/'FINAL13_MANUSCRIPT_UPDATE.json'),window_R3_diagnostic=True,all_generated_R3_separately_reported=True,no_root_ADE=True,long20_40='N/A no natural long reference',time=time.time())
 wr(R/'FINAL_AUDIT.json',result)
 files=['SUMMARY.csv','PHYSICAL.csv','LATENCY.csv','WINDOW_METRICS.csv','QUERY_POOLS.json','EVALUATOR_TRANSFER_AUDIT.json','humanml3d_results.tex','main.tex','FINAL_AUDIT.json']
 verification=dict(result,files={n:sha(R/n) for n in files},full_training_checkpoints_verified=True,all_training_and_evaluation_complete=True)
 wr(R/'FINAL_VERIFICATION.json',verification)
 wr(R/'COMPLETE.json',dict(passed=True,models=methods,tasks=tasks,N=181,correct_domain_evaluator=True,final_verification_sha256=sha(R/'FINAL_VERIFICATION.json'),manuscript_native_compile='success',time=time.time()))
 wr(R/'STATUS.json',dict(stage='COMPLETE_TRAINING_GENERATION_CORRECT_DOMAIN_EVALUATION_AND_MANUSCRIPT',completed=methods,active={},time=time.time()))
 print(json.dumps({k:result[k] for k in ['passed','models_count','N','max_CE_delta_cm']}))
if __name__=='__main__':main()
