"""Seal and drive the authorized release through existing bounded clip batches."""
from motion_mamba2.paths import artifact_path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,collections,subprocess,sys,time,fcntl,shutil
from pathlib import Path
from g1_pipeline.common import ROOT,sha,write_json,write_jsonl,atomic_bytes,read_jsonl,now
from g1_pipeline.kinematic_policy import BASE,load
REQUEST=Path(artifact_path('9_17-_md_65267f5053'))

def policy():
 p=BASE/'execution_policy.json'
 if p.exists():return load(p)
 value={'scope':'kinematic_training','frozen_solver':'v25','request_path':str(REQUEST),'request_sha256':sha(REQUEST),'aggregate_retention_min':None,'batch_rejection_rate_stop':None,'dynamic_required':False,'dynamic_validation':'NOT_RUN','maximum_hinge_correction_rad':.35,'qa_threshold_source':'configs/gates.v2.yaml + configs/bones_core_refinement_v24.json','original_hard_rejection_auto_reversal':False,'flat_core_hard_physics':True,'complex_contact_diagnostic':True,'self_collision_exemptions_added':False,'uncertain_individuals':'REVIEW_PENDING','heldout_calibration_in_train':False,'original_split_preserved':True,'formal_training_authorized':False,'initial_workers':16,'renderer_workers':1,'first1000_is_operational_window':True,'no_source_yield_gate':True}
 write_json(p,value);return value

def seal(extra_panels=None):
 policy();out=BASE/'execution_release.json'
 checks=load(BASE/'policy_branch_checks.json')
 if checks['status']!='PASS' or checks['policy_module_sha256']!=sha(ROOT/'g1_pipeline/kinematic_policy.py') or checks['policy_sha256']!=sha(BASE/'execution_policy.json'):raise ValueError('ADMISSION_BRANCH_CHECKS_STALE')
 if load(BASE/'entry_checks.json')['status']!='PASS':raise ValueError('REAL_ENTRY_CHECK_REQUIRED')
 if load(BASE/'hiphi_entry_checks.json')['status']!='PASS':raise ValueError('STAGED_HIPHI_ENTRY_CHECK_REQUIRED')
 if out.exists():return load(out)
 inputs=load(BASE/'input_release.json')
 for path,bound in inputs['input_bindings'].items():
  if sha(ROOT/path)!=bound:raise ValueError('INPUT_RELEASE_CHANGED:'+path)
 from g1_pipeline.calibration_protocol import require_frozen,PATH
 frozen=require_frozen();bindings=dict(frozen['bindings']);bindings[str(PATH.relative_to(ROOT))]=sha(PATH)
 # Existing historical solver/loader/config bindings are reused verbatim.
 code={}
 for short in ['hml','hiphi']:
  ns=load(ROOT/f'audit/current_pilot_{short}100_run.json')['namespace']
  code.update(load(ROOT/'work'/ns/'run.json')['fingerprint']['code'])
 for name,bound in code.items():bindings['g1_pipeline/'+name+'.py']=bound
 for path in ['g1_pipeline/parallel_runner.py','g1_pipeline/parallel_clip.py','g1_pipeline/batch_state.py','g1_pipeline/kinematic_runner.py','g1_pipeline/kinematic_clip.py','g1_pipeline/kinematic_policy.py','scripts/kinematic_release.py','scripts/kinematic_prepare.py',str((BASE/'execution_policy.json').relative_to(ROOT)),str((BASE/'storage.json').relative_to(ROOT)),str((BASE/'input_release.json').relative_to(ROOT)),str((BASE/'prior_pilot_outputs.json').relative_to(ROOT))]:bindings[path]=sha(ROOT/path)
 from g1_pipeline import locked_target
 olddir,oldwrite=locked_target.ASSET_DIR,locked_target.write_json
 locked_target.ASSET_DIR=BASE/'asset';locked_target.write_json=lambda p,v:write_json(BASE/'asset_equivalence.json',v)
 try:xml,asset_report=locked_target.prepare()
 finally:locked_target.ASSET_DIR,locked_target.write_json=olddir,oldwrite
 bindings.update(asset_report['source']['resolved_meshes']);bindings[asset_report['source']['xml']]=asset_report['source']['xml_hash']
 protected={}
 for p in list((ROOT/'gates').glob('*.json'))+list((ROOT/'gates/pilot').glob('*.json'))+[ROOT/'release_manifest.json',ROOT/'audit/current_work_state.json',ROOT/'audit/pilot_final_evidence_audit.json',ROOT/'audit/bones_frozen_accept_v25_final_evidence_audit.json']:
  protected[str(p.relative_to(ROOT))]=sha(p)
 panels={p['path']:p['sha256'] for p in inputs['panels']};panels.update(extra_panels or {})
 value={'time':now(),'scope':'kinematic_training','status':'SEALED','bindings':bindings,'protected_history':protected,'original_clip_code':code,'asset':{'path':str(xml),'sha256':sha(xml),'report':asset_report},'panel_hashes':panels,'input_release_sha256':sha(BASE/'input_release.json'),'execution_policy_sha256':sha(BASE/'execution_policy.json'),'original_calibration_binding_used_for_admission':False}
 write_json(out,value);return value

def check_batch(namespace):
 base=ROOT/'work'/namespace;design=load(base/'run.json');tasks=load(ROOT/design['panel'])['tasks'];rows=read_jsonl(base/'results.jsonl')
 if len(rows)!=len(tasks) or [r['sample_id'] for r in rows]!=[t['row']['sample_id'] for t in tasks]:raise ValueError('BATCH_LEDGER_RECONCILIATION_FAILED')
 counts=collections.Counter();source=collections.defaultdict(collections.Counter);fatal=[]
 for r in rows:
  decision=r.get('decision', 'ERROR' if r['status']=='ERROR' else 'REJECTED')
  counts[decision]+=1;source[r['dataset']][decision]+=1
  error=r.get('error','')
  if any(token in error for token in ('CHANGED','MISMATCH','No space left','MemoryError','ARTIFACT_COMMIT','Input/output error')):fatal.append({'sample_id':r['sample_id'],'error':error})
  if decision=='ACCEPTED_KINEMATIC':
   c=load(ROOT/r['output_paths']['contract_path']);d=load(ROOT/r['decision_path'])
   if c['status']!='PASS' or d['reasons']:raise ValueError('INVALID_ACCEPTED_DATA')
   for path,bound in r['output_hashes'].items():
    if sha(ROOT/path)!=bound:raise ValueError('OUTPUT_HASH_CHANGED')
 if fatal:raise ValueError('SYSTEMATIC_EXECUTION_FAULT:'+repr(fatal[:4]))
 summary=load(base/'summary.json')
 import sqlite3,numpy as np
 db=sqlite3.connect('file:'+str(base/'state.sqlite')+'?mode=ro',uri=True)
 execution=[]
 for encoded, in db.execute('select output_json from items'):
  output=json.loads(encoded);execution.append(load(Path(output['execution_path'])))
 db.close()
 service=[v.get('wall_sec',0) for v in execution if not v.get('reused')]
 resource={'wall_sec':summary['wall_sec'],'fresh_output_frames':sum(r.get('frames',0) for r in rows if not r.get('reused')),'reused_output_frames':sum(r.get('frames',0) for r in rows if r.get('reused')),'fresh_only_run':not any(r.get('reused') for r in rows) and summary['throughput_valid'],'cpu_seconds_sum':sum(v.get('cpu_sec',0) for v in execution),'peak_sampled_worker_rss_bytes':summary['peak_sampled_worker_rss_bytes'],'io_read_blocks_512_sum':sum(v.get('read_blocks',0) for v in execution),'io_write_blocks_512_sum':sum(v.get('write_blocks',0) for v in execution),'clip_service_p50_sec':float(np.percentile(service,50)) if service else None,'clip_service_p95_sec':float(np.percentile(service,95)) if service else None,'clip_service_max_sec':max(service) if service else None,'active_worker_samples':summary['samples'],'source_effective_output_frames_per_sec_in_mixed_batch':{source:sum(r.get('frames',0) for r in rows if r['dataset']==source and not r.get('reused'))/summary['wall_sec'] for source in ['bones','humanml3d','hiphi']},'render_included':False,'review_included':False,'timing_limitation':'Mixed source batch contribution rates, not equal-cost source benchmarks or linear worker scaling; reused outputs excluded from new-compute frames.'}
 resource['accepted_frames_per_sec']=sum(r.get('frames',0) for r in rows if r.get('decision')=='ACCEPTED_KINEMATIC' and not r.get('reused'))/summary['wall_sec']
 write_json(base/'full_queue_resources.json',resource)
 value={'time':now(),'status':'OPERATIONALLY_VALID','namespace':namespace,'clips':len(rows),'counts':dict(counts),'sources':{s:dict(v) for s,v in source.items()},'yield_threshold_applied':False,'frames':sum(r.get('frames',0) for r in rows),'accepted_frames':sum(r.get('frames',0) for r in rows if r.get('decision')=='ACCEPTED_KINEMATIC'),'reused_clips':sum(bool(r.get('reused')) for r in rows),'summary_sha256':sha(base/'summary.json'),'ledger_sha256':sha(base/'results.jsonl')}
 write_json(base/'operational_check.json',value);return value

def run():
 spec=seal();inputs=load(BASE/'input_release.json');BASE.mkdir(exist_ok=True)
 storage=load(BASE/'storage.json');work_storage=Path(storage['work_storage']);work_storage.mkdir(parents=True,exist_ok=True)
 with (BASE/'release.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('KINEMATIC_RELEASE_ALREADY_RUNNING')
  completed=[];began=time.monotonic()
  for i,panel in enumerate(inputs['panels']):
   namespace='kinematic_training_20260917_v1_batch_'+f'{i:04d}';base=ROOT/'work'/namespace
   target=work_storage/namespace
   if not base.exists():
    target.mkdir(exist_ok=True);base.symlink_to(target,target_is_directory=True)
   if not base.is_symlink() or base.resolve()!=target.resolve():raise ValueError('RELEASE_STORAGE_BINDING_MISMATCH')
   if (base/'operational_check.json').exists():
    old=load(base/'operational_check.json')
    if sha(base/'results.jsonl')!=old['ledger_sha256']:raise ValueError('COMPLETED_LEDGER_CHANGED')
    completed.append(old);continue
   if (BASE/'STOP').exists():
    write_json(BASE/'status.json',{'time':now(),'phase':'STOPPED_AT_BATCH_BOUNDARY','completed_batches':len(completed)});return
   if min(shutil.disk_usage(BASE).free,shutil.disk_usage(work_storage).free)<20*2**30:raise RuntimeError('RESOURCE_DISK_FREE_BELOW_20_GIB')
   command=[sys.executable,'-m','g1_pipeline.kinematic_runner','run','--namespace',namespace,'--panel',panel['path'],'--workers','16']
   if (base/'run.json').exists():command.append('--resume')
   log=(BASE/'execution.log').open('ab')
   proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
   while proc.poll() is None:
    heartbeat=load(base/'heartbeat.json') if (base/'heartbeat.json').exists() else {}
    write_json(BASE/'status.json',{'time':now(),'phase':'FULL_CONVERSION','pid':os.getpid(),'coordinator_pid':proc.pid,'namespace':namespace,'batch_index':i,'batch_total':len(inputs['panels']),'completed_batches':len(completed),'completed_clips':sum(v['clips'] for v in completed),'completed_frames':sum(v['frames'] for v in completed),'accepted_frames':sum(v['accepted_frames'] for v in completed),'current_batch':heartbeat,'elapsed_sec':time.monotonic()-began,'ready':False})
    time.sleep(15)
   log.close()
   if proc.returncode:raise RuntimeError('BATCH_PROCESS_FAILED:'+namespace)
   result=check_batch(namespace);completed.append(result)
   write_json(BASE/'batch_checks.json',{'time':now(),'completed':completed,'first1000_does_not_require_quality_yield':True})
   print(json.dumps({'batch':i,'clips':result['clips'],'counts':result['counts'],'frames':result['frames'],'auto_continue':True}),flush=True)
  for path,bound in spec['protected_history'].items():
   if sha(ROOT/path)!=bound:raise ValueError('HISTORICAL_EVIDENCE_CHANGED:'+path)
  write_json(BASE/'status.json',{'time':now(),'phase':'CONVERSION_COMPLETE_FINALIZATION_REQUIRED','completed_batches':len(completed),'completed_clips':sum(v['clips'] for v in completed),'completed_frames':sum(v['frames'] for v in completed),'accepted_frames':sum(v['accepted_frames'] for v in completed),'ready':False})
  # Final manifests/stats/model smoke are separate real checks, never inferred from conversion.

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['policy','seal','run','status']);a=p.parse_args()
 if a.command=='policy':print(policy())
 elif a.command=='seal':print(seal()['status'])
 elif a.command=='status':print(load(BASE/'status.json') if (BASE/'status.json').exists() else load(BASE/'preparation_status.json'))
 else:run()
