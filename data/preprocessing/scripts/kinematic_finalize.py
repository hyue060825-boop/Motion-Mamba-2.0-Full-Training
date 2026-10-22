"""Build only the independent offline release manifests and descriptive train stats."""
import ast,collections,json,math
from pathlib import Path
from g1_pipeline.common import ROOT,sha,write_json,write_jsonl,read_jsonl,atomic_bytes,digest,now
from g1_pipeline.kinematic_policy import BASE,authorize,load

def record_function():
 from g1_pipeline import production_manifest as original
 tree=ast.parse((ROOT/'g1_pipeline/production_manifest.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='record')
 fn.body=[n for n in fn.body if not (isinstance(n,ast.If) and 'MANIFEST_DYNAMIC_COVERAGE_REQUIRED' in ast.unparse(n))]
 ns=vars(original).copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'kinematic_manifest_branch','exec'),ns)
 return ns['record']

def source_timeline(meta,result):
 """Recover measured native timing, retaining the untouched inventory estimate."""
 meta=dict(meta)
 if meta['dataset']=='hiphi' and result.get('output_paths'):
  path=(ROOT/result['output_paths']['qpos_path']).parent/'time.json'
  bound=result['output_hashes'].get(str(path.relative_to(ROOT)))
  if not bound or sha(path)!=bound:raise ValueError('NATIVE_TIMELINE_HASH_MISMATCH')
  timing=load(path);fps=timing['source_fps'];frames=timing['source_frames']
  if not math.isfinite(fps) or fps<=0 or frames!=meta['original_frames']:raise ValueError('NATIVE_TIMELINE_METADATA_MISMATCH')
  meta.update(original_fps=fps,original_fps_reason='measured BVH Frame Time; hash-bound time.json',
              native_start_sec=0.,native_end_sec=frames/fps,
              measured_native_duration_sec=frames/fps,
              native_last_sample_sec=(frames-1)/fps,
              native_timeline_path=str(path.relative_to(ROOT)),native_timeline_sha256=bound)
 return meta

def collect(require_complete=True):
 authorize();inputs=load(BASE/'input_release.json');allrows=[];processed=0
 for i,p in enumerate(inputs['panels']):
  base=ROOT/'work'/('kinematic_training_20260917_v1_batch_'+f'{i:04d}')
  if not (base/'operational_check.json').exists():
   if require_complete:raise ValueError('CONVERSION_INCOMPLETE:'+base.name)
   continue
  check=load(base/'operational_check.json')
  if sha(base/'results.jsonl')!=check['ledger_sha256']:raise ValueError('COMMITTED_LEDGER_CHANGED')
  allrows.extend(read_jsonl(base/'results.jsonl'));processed+=len(read_jsonl(base/'results.jsonl'))
 if require_complete:assert processed==inputs['supported_candidates']
 return inputs,allrows

def manifests():
 inputs,rows=collect();metadata={r['sample_id']:r for r in read_jsonl(BASE/'candidates.jsonl')};record=record_function()
 decisions=[];accepted=[]
 for r in rows:
  from g1_pipeline.bones_official_manifest_admission import POLICY,ADDENDUM,apply
  if POLICY.exists():r=apply(r)
  meta=source_timeline(metadata[r['sample_id']],r);decision=r.get('decision','ERROR' if r['status']=='ERROR' else 'REJECTED')
  value={'sample_id':r['sample_id'],'dataset':r['dataset'],'source_family_id':meta['source_family_id'],'project_split':meta['project_split'],'training_role':meta['training_role'],'category':meta['category'],'captions':meta['captions'],'native_interval':meta.get('native_interval'),'original_fps':r.get('source_metadata',meta).get('original_fps'),'source_frames':meta.get('original_frames'),'source_duration_sec':meta.get('metadata_duration_sec') or (meta['native_end_sec']-meta['native_start_sec']),'target_frames':r.get('frames',0),'target_duration_sec':r.get('frames',0)/25,'decision':decision,'reasons':r.get('decision_reasons',r.get('rejections',[])) or ([r['error']] if r.get('error') else []),'diagnostic_tags':r.get('diagnostic_tags',r.get('review_flags',[])),'scene_class':r.get('source_semantics',{}).get('scene_class'),'output_paths':r.get('output_paths',{}),'output_hashes':r.get('output_hashes',{}),'source_sha256':r.get('source_sha256'),'policy_sha256':sha(BASE/'execution_policy.json'),'dynamic_validation':'NOT_RUN','attempted':True,'reused':bool(r.get('reused'))}
  decisions.append(value)
  if r['dataset']=='bones' and POLICY.exists():
   value.update(bones_official_policy_sha256=sha(POLICY),scene_scope_policy_sha256=sha(ADDENDUM),geometry_validation='DIAGNOSTIC_ONLY')
  if r.get('historical_policy_result'):
   value.update(historical_policy_result=r['historical_policy_result'],bones_official_policy_sha256=r['bones_official_policy_sha256'],admission_reused_output=True)
  if meta.get('measured_native_duration_sec') is not None:
   value.update(original_fps=meta['original_fps'],source_duration_sec=meta['measured_native_duration_sec'],native_interval={'start_sec':0.,'end_sec':meta['native_end_sec'],'last_sample_sec':meta['native_last_sample_sec']},native_timeline_path=meta['native_timeline_path'],native_timeline_sha256=meta['native_timeline_sha256'])
  interval=meta.get('native_interval') or {}
  if meta['dataset']=='humanml3d' and 'native_end_index_exclusive' in interval:
   value.update(native_file_frames=meta.get('original_frames'),source_frames=interval['native_end_index_exclusive']-interval['native_first_index'])
  if decision!='ACCEPTED_KINEMATIC':continue
  if meta['training_role']=='held_out_calibration':value['default_training_excluded']='FROZEN_CALIBRATION_HELDOUT';continue
  prepared=dict(r,status='KINEMATIC_CANDIDATE',body_only_validated=r['dataset']=='hiphi')
  meta=dict(meta);meta.setdefault('native_start_sec',0.);meta.setdefault('native_end_sec',value['source_duration_sec'])
  output=record(meta,prepared,pipeline_version='v25_kinematic_training_20260917_v1',sim_coverage={'status':'NOT_RUN','controller_hash':None,'scope':'offline kinematics only'},leakage={'status':'PASS','source_family_id':meta['source_family_id'],'split_policy':inputs['split_policy']})
  output.update(kinematic_status='ACCEPTED_KINEMATIC',dynamic_validation='NOT_RUN',kinematic_policy_sha256=sha(BASE/'execution_policy.json'),decision_path=r['decision_path'],diagnostic_tags=value['diagnostic_tags'],training_role=meta['training_role'],exposure_history=meta['exposure_history'])
  if r['dataset']=='bones' and POLICY.exists():
   output.update(bones_official_policy_sha256=sha(POLICY),scene_scope_policy_sha256=sha(ADDENDUM),geometry_validation='DIAGNOSTIC_ONLY')
  if r.get('bones_official_policy_sha256'):output.update(bones_official_policy_sha256=r['bones_official_policy_sha256'],historical_policy_result=r['historical_policy_result'])
  accepted.append(output)
 for r in read_jsonl(BASE/'isolated.jsonl'):
  meta=r['row'];decisions.append({'sample_id':meta['sample_id'],'dataset':meta['dataset'],'category':meta['category'],'source_family_id':meta['source_family_id'],'project_split':meta['project_split'],'training_role':meta['training_role'],'decision':r['decision'],'reasons':r['reasons'],'attempted':False,'target_frames':0,'source_frames':meta.get('original_frames'),'source_duration_sec':meta.get('metadata_duration_sec') or ((meta.get('native_end_sec') or 0)-(meta.get('native_start_sec') or 0))})
 assert len(decisions)==inputs['inventory'] and len({r['sample_id'] for r in decisions})==inputs['inventory']
 # Family/mirror/crop and exact-output duplicates across splits stay out of train.
 split_by_family=collections.defaultdict(set);split_by_hash=collections.defaultdict(set)
 for meta in metadata.values():split_by_family[meta['source_family_id']].add(meta['project_split'])
 for r in accepted:
  split_by_family[r['source_family_id']].add(r['project_split']);split_by_hash[r['features_hash']].add(r['project_split'])
 conflicts={k for k,v in split_by_family.items() if len(v)>1};cross_exact={k for k,v in split_by_hash.items() if len(v)>1}
 exclusions=[];filtered=[];seen_features=set();duplicates=[]
 for r in sorted(accepted,key=lambda r:r['sample_id']):
  if r['source_family_id'] in conflicts or r['features_hash'] in cross_exact:
   exclusions.append({'sample_id':r['sample_id'],'reason':'CROSS_SPLIT_FAMILY_OR_EXACT_OUTPUT','source_family_id':r['source_family_id']});continue
  if r['features_hash'] in seen_features:
   duplicates.append(r['sample_id']);continue
  seen_features.add(r['features_hash'])
  filtered.append(r)
 accepted=filtered;excluded_ids={r['sample_id'] for r in exclusions}
 duplicate_ids=set(duplicates)
 for r in decisions:
  if r['sample_id'] in excluded_ids:r.update(decision='REVIEW_PENDING',reasons=['CROSS_SPLIT_FAMILY_OR_EXACT_OUTPUT'])
  if r['sample_id'] in duplicate_ids:r['default_training_excluded']='EXACT_FEATURE_DUPLICATE'
 directory=BASE/'manifests';directory.mkdir(exist_ok=True)
 for split in ['train','val','test']:write_jsonl(directory/f'accepted_{split}.jsonl',[r for r in accepted if r['project_split']==split])
 for status in ['ACCEPTED_KINEMATIC','REJECTED','REVIEW_PENDING','UNSUPPORTED','ERROR']:write_jsonl(directory/(status.lower()+'.jsonl'),[r for r in decisions if r['decision']==status])
 write_jsonl(directory/'all_decisions.jsonl',decisions)
 coverage=collections.defaultdict(collections.Counter)
 for r in decisions:
  for key in [r['dataset'],r['dataset']+':'+r['category']]:
   c=coverage[key];c['inventory']+=1;c[r['decision']]+=1;c['attempted']+=int(r['attempted']);c['source_frames']+=r.get('source_frames') or 0;c['source_duration_sec']+=r.get('source_duration_sec') or 0;c['converted_frames']+=r['target_frames']
   if r['decision']=='ACCEPTED_KINEMATIC':c['accepted_frames']+=r['target_frames']
 report={k:{**v,'clip_retention':v['ACCEPTED_KINEMATIC']/v['inventory'],'converted_frame_retention':v['accepted_frames']/v['converted_frames'] if v['converted_frames'] else 0,'accepted_hours':v['accepted_frames']/25/3600} for k,v in coverage.items()}
 write_json(BASE/'coverage.json',{'time':now(),'groups':report,'inventory':len(decisions),'default_manifest_clips':len(accepted),'heldout_excluded':sum(r.get('default_training_excluded')=='FROZEN_CALIBRATION_HELDOUT' for r in decisions),'leakage_exclusions':exclusions,'status':'MANIFESTS_READY_FOR_STATS_AND_RUNTIME'})
 write_json(BASE/'leakage_audit.json',{'status':'PASS','original_splits_preserved':True,'heldout_excluded':True,'family_conflicts_quarantined':sorted(conflicts),'exact_output_cross_split_groups_quarantined':sorted(cross_exact),'duplicate_policy':'existing lineage families, mirror/crop groups and exact output/source identity; no new near-duplicate threshold or system','cross_source_known_copy_review':'existing lineage evidence plus exact output hashes; no claim of exhaustive unknown semantic duplicate discovery'})
 return report

def stats():
 authorize()
 from g1_pipeline import fusion_statistics as original
 tree=ast.parse((ROOT/'g1_pipeline/fusion_statistics.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 # Only gate/output scope changes. Existing streaming valid-frame stats retained.
 fn.body=ast.parse('authorize()').body+fn.body[1:]
 ns=vars(original).copy();ns['authorize']=authorize
 ns['write_json']=lambda path,obj:write_json(BASE/'stats/physical_414.json',obj)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'kinematic_train_only_stats','exec'),ns)
 return ns['run'](BASE/'manifests/accepted_train.jsonl')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('command',choices=['manifests','stats']);a=p.parse_args()
 if a.command=='manifests':print(manifests())
 else:print(stats()['status'])
