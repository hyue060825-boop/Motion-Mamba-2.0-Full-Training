"""Final read-only verification after metrics and physical diagnostics complete."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,time
from util import ROOT,read,write,sha

def main():
 n=read(ROOT/'NUMERICS_COMPLETE.json');c=read(ROOT/'COMPARISON_COMPLETE.json');p=read(ROOT/'PHYSICAL_COMPLETE.json')
 assert n['new_trajectories']==c['new_native100_trajectories']==1518
 for name,h in n['result_hashes'].items():assert sha(ROOT/'aggregate'/name)==h,name
 for doc in [c,p]:
  for name,h in doc['result_hashes'].items():assert sha(ROOT/'comparison/aggregate'/name)==h,name
 for name,h in read(ROOT/'PLAN.json')['execution_source_hashes'].items():assert sha(ROOT/name)==h,name
 cp=Path(read(ROOT/'CHECKPOINTS.json')['native100']['checkpoint'])
 assert sha(cp)=='d21b9a22616fcfa2244629d1d4176a6de0c4f567cc5c69022458321afea0c9ca'
 proof=dict(status='PASS_FIVE_MODEL_FIRST_EXPERIMENT_NUMERICS_AND_NATIVE100K_IDENTITY',
   new_trajectories=1518,source_count=506,training_updates=0,simulation_changes=False,
   corrected_ardy100k_used=True,official_motion_mamba2_not_mixed_into_requested_models=True,
   baseline_names=c['models'],native100_checkpoint=str(cp),native100_sha256=sha(cp),
   numerics_sha256=sha(ROOT/'NUMERICS_COMPLETE.json'),comparability_sha256=sha(ROOT/'comparison/COMPARABILITY_AUDIT.json'),
   inference_sha256=sha(ROOT/'INFERENCE_EXECUTION_AUDIT.json'),physical_sha256=sha(ROOT/'PHYSICAL_COMPLETE.json'),
   report_pending_visual_QA=True,time=time.time())
 write(ROOT/'FINAL_VERIFICATION.json',proof)
 plan=read(ROOT/'PLAN.json')
 write(ROOT/'EFFECTIVE_PROTOCOL.json',dict(status='COMPLETE_EFFECTIVE_PROTOCOL_CLARIFICATION',
  models=c['models'],new_native100_trajectories=1518,sources_per_condition=506,actors=22,
  actual_tasks=plan['tasks'],sampling_steps=10,inference_batch=2,frame_rate=25,window_frames=52,
  prefix_frames=24,goal_visible_from_beginning=True,goal_channels='root XZ only',
  same_spatial_goal_across_deadlines=True,goal_within_window_frame_zero_based=51,
  endpoint_only_score=True,intermediate_path_penalty=False,root_ADE_reported=False,
  all_window_quality='506 local52 generated/reference windows per window; 319 unique-caption R@3 queries, 32 candidates',
  encoder_seeds=[1701,2903],main_encoder_seed=1701,retrieval_pool_seeds=[4101,4102,4103],
  bootstrap='actor then source, 2000 draws; CE and R@3 only, no calibrated FID CI',
  source_exposure='Known generator-training HiPHI sources; post-hoc controlled diagnosis',
  metadata_note='PLAN preserves inherited historical proposal text in phase_design and source_protocol. Actual execution is the three aligned tasks above, not the four proposed phase interventions. Per-window local52 scoring was executed for this 506-source cohort; it is not the separate 1024-source study.',
  execution_audit_sha256=proof['inference_sha256'],comparability_sha256=proof['comparability_sha256'],time=time.time()))
 old=read(Path(artifact_path('checkpoints_json_798bfdca7e')))
 fixed=read(Path(artifact_path('checkpoints_json_9802580e14')))
 catalog={m:old[m] for m in ['minimal','native110','native117']}
 catalog.update(read(ROOT/'CHECKPOINTS.json'));catalog.update(fixed)
 write(ROOT/'COMPARISON_CHECKPOINTS.json',{m:catalog[m] for m in c['models']})
 write(ROOT/'STATUS.json',dict(status='COMPLETE_FIVE_MODEL_NUMERICS_PENDING_LOCAL_PUBLICATION_QA',
    new_native100_trajectories=1518,models=c['models'],training_updates=0,time=time.time()))
 print('FINAL_NUMERICS_VERIFIED')

if __name__=='__main__':main()
