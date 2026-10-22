from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,shutil,time,hashlib,os
R=Path(artifact_path('g1_external_matched100k_evaluation_20261019_aa0c66c5b2'))
C=Path(artifact_path('transformer_reference100k_corrected_evaluation_20261017_7a91e02b7e'))
B=Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d'))
S=Path(artifact_path('g1_external_matched100k_20261019_f77433d79c'))
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert not (R/'PLAN.json').exists(),'Do not overwrite frozen evaluation'
for n,source in [('panels',B/'panels'),('manifest.jsonl',B/'manifest.jsonl'),('text_cache',B/'text_cache'),('source_protocol',B/'source_protocol')]:
 (R/n).symlink_to(source,target_is_directory=source.is_dir())
for n in ['run_job.py','controller.py','score_job.py','aggregate_reference.py','committed_rolling.py','util.py']:
 shutil.copy2(C/n,R/n)
shutil.copytree(C/'sampler',R/'sampler',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(R/'checkpoint_loader.py',R/'sampler/checkpoint_loader.py')
p=R/'util.py';t=p.read_text().replace(str(C),str(R)).replace("MODELS=['ardy']","MODELS=['dip','dart']").replace("TASKS=['aligned_w04','aligned_w09','aligned_w19']","TASKS="+repr(list(read(B/'PLAN.json')['tasks']))) ;p.write_text(t)
p=R/'run_job.py';t=p.read_text().replace("mode={'ardy':'original'}[a.model]","mode='original'").replace('actual_target_exposed_before_entering_current_window=True','global_request_includes_target_from_start=True,denoiser_visibility_see_INFERENCE_TRACE=True').replace('sampler_unchanged=True','sampler_unchanged=False,shared_committed_sampler=True,fixed_context_baseline=True');p.write_text(t)
p=R/'committed_rolling.py';t=p.read_text().replace("mm=batch['motion_mask'][:,start:].clone();obs=batch['observed_motion'][:,start:].clone()", "capacity_end=min(T,start+248)\n  mm=batch['motion_mask'][:,start:capacity_end].clone();obs=batch['observed_motion'][:,start:capacity_end].clone()")
t=t.replace('0,h,T-start,text','0,h,capacity_end-start,text').replace('total=T-start,goal_visible=bool(mm.count_nonzero())','total=T-start,actual_input_frames=capacity_end-start,future_capacity_frames=capacity_end-start-h-52,goal_visible=bool(mm.count_nonzero())').replace('ACTUAL_COMMITTED_V3_EXPLICIT_HISTORY_196_AND_DECODER_52','ACTUAL_FIXED_CAPACITY_248_HISTORY_196_COMMITTED_DECODER_52')
p.write_text(t)
p=R/'sampler/evaluate_motion_mamba2_official_table5.py';t=p.read_text().replace('raw.root_model.eval()\n    raw.body_model.eval()','raw.eval()').replace('list(raw.root_model.parameters()) + list(raw.body_model.parameters())','list(raw.parameters())');p.write_text(t)
p=R/'controller.py';t=p.read_text().replace("g['card'] in [4,5,6,7]","g['card'] in list(range(8))").replace('unused GPUs4-7','unused GPUs0-7').replace('N_new_trajectories=1518','N_new_trajectories=7084').replace('FORMAL_CONTROLLED_OFFICIAL_MOTION_MAMBA2_EVALUATION','FORMAL_FIXED_CONTEXT_G1_BASELINES_EVALUATION').replace('ALL_1518_OFFICIAL_TRAJECTORIES_AND_LOCAL52_FEATURES_COMPLETE','ALL_7084_BASELINE_TRAJECTORIES_AND_LOCAL52_FEATURES_COMPLETE')
# This run has its own comparison writer, not the old hard-coded ARDY repair.
start=t.index("    with (ROOT/'combine.log').open('a') as log:")
end=t.index("    write(ROOT/'STATUS.json'",start)
t=t[:start]+t[end:];p.write_text(t)
plan=read(B/'PLAN.json');plan.update(created=time.time(),models={'dip':'DiP-G1 common-codec architecture control','dart':'DART-MLP-G1 common-codec architecture control'},cards=list(range(8)),purpose='Evaluate frozen latest100k controls, all seven paired506-source conditions',
 maximum_total_input_frames=248,full_future_conditioning_equivalent=False,comparison_scope='same global tasks, each method native input capacity; not an equal-full-future information ablation',
 main_table_excludes=['Minimal'],weights_or_training_changed=False)
info={}
for m,n in [('dip','DiP_G1'),('dart','DART_MLP_G1')]:
 latest=read(S/n/'LATEST.json');assert latest['step']==100000 and latest['complete_Adam'] and latest['rank_states_equal'] and latest['four_rank_RNG']
 ck=Path(latest['checkpoint']);st=ck.stat();info[m]=dict(training_root=str(S/n),checkpoint=str(ck),sha256=sha(ck),stats_dir=read(S/n/'configs/batch512.json')['stats_dir'],step=100000,exposures=latest['exposures'],frozen_file_stat=dict(size=st.st_size,mtime_ns=st.st_mtime_ns))
write(R/'CHECKPOINTS.json',info)
plan['prepared_hashes']={n:sha(R/n) for n in ['manifest.jsonl','CHECKPOINTS.json','sampler/evaluate_motion_mamba2_official_table5.py']}
plan['execution_source_hashes']={n:sha(R/n) for n in ['controller.py','run_job.py','score_job.py','checkpoint_loader.py','sampler/checkpoint_loader.py','committed_rolling.py','official_setup.py','aggregate.py','aggregate_reference.py','util.py']}
write(R/'PLAN.json',plan)
write(R/'PREPARED.json',dict(status='PASS_CPU_COHORT_GOALS_SEEDS_AND_OFFICIAL_RELEASE',plan_sha256=sha(R/'PLAN.json'),parent_plan_sha256=sha(B/'PLAN.json'),no_new_sample_selection=True))
print(json.dumps(dict(root=str(R),checkpoints=info,tasks=list(plan['tasks']),jobs=224),indent=2))
