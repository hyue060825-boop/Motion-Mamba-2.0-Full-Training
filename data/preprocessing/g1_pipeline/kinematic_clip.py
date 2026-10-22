"""Prepare one authorized full clip, reuse v25 computation, and apply offline QA."""
import ast,copy,importlib,json,time
from pathlib import Path
from .common import ROOT,HML,sha,write_json
from .kinematic_policy import BASE,authorize,decide,load

def compile_authorized_process():
 from . import source_motion_processing as processing
 tree=ast.parse((ROOT/'g1_pipeline/source_motion_processing.py').read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process')
 assert isinstance(fn.body[1],ast.If) and ast.unparse(fn.body[1].test)=='pilot'
 fn.body[:2]=ast.parse('authorize(dataset)').body
 processing.authorize=authorize
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ROOT/'g1_pipeline/source_motion_processing.py'),'exec'),vars(processing))
 for name in ('production_hml','production_hiphi'):importlib.import_module('g1_pipeline.'+name).process=processing.process


def execute_original(task,out,asset,expected_code):
 from . import parallel_clip
 tree=ast.parse((ROOT/'g1_pipeline/parallel_clip.py').read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
 class PackageBinding(ast.NodeTransformer):
  def visit_keyword(self,node):
   if node.arg=='packagepath':node.value=ast.parse("ROOT/task['package_path']",mode='eval').body
   return node
 fn=PackageBinding().visit(fn);ns=vars(parallel_clip).copy()
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ROOT/'g1_pipeline/parallel_clip.py'),'exec'),ns)
 return ns['execute'](task,out,asset,expected_code)


def bones(task,out):
 import numpy as np
 from .robot import compile_target,read_csv,FORMAT
 from .resample import qpos as resample
 from .production_contract import export_and_check
 from .bones_pair_qa import replay
 from .bones_continuity_qa import metrics as continuity
 from .scene_policy import classify
 from .bones_relative_motion_v2 import labels
 row=task['row'];native=Path(row['sealed_native_path'])
 if sha(native)!=row['sealed_native_sha256']:raise ValueError('SEALED_NATIVE_INPUT_CHANGED')
 m,_=compile_target();q,t,_=read_csv(native,format_id=FORMAT,fps=row['original_fps'])
 if len(q)!=row['original_frames']:raise ValueError('BONES_NATIVE_FRAME_METADATA_CONFLICT')
 semantics=classify(row);semantics['action_labels']=labels(row)
 # Official source retains its complete native support; no candidate trajectory.
 try:q,t,tmap=resample(q,t,m.jnt_range[1:])
 except ValueError as exc:
  if str(exc)=='SOURCE_HINGE_LIMIT':return {'sample_id':row['sample_id'],'dataset':'bones','status':'REJECTED','source_metadata':row,'source_semantics':semantics,'source_sha256':row['sealed_native_sha256'],'rejections':['SOURCE_HINGE_LIMIT'],'review_flags':[],'decision':'REJECTED','decision_reasons':['SOURCE_HINGE_LIMIT'],'frames':0},{}
  raise
 c,paths,hashes=export_and_check(out/'motion',row['sample_id'],q,t,{'source':'BONES official G1 main pool','native_path':str(native),'native_sha256':row['sealed_native_sha256'],'resample_count':tmap['resample_count']},row['captions'])
 metrics={'geometry':replay(m,q)[2],'continuity':continuity(m,q,t),'official_source':True,'rejections':[],'review_flags':[]}
 write_json(out/'motion/time.json',tmap);write_json(out/'motion/solver.json',metrics)
 for name in ['time.json','solver.json']:hashes[str((out/'motion'/name).relative_to(ROOT))]=sha(out/'motion'/name)
 return {'sample_id':row['sample_id'],'dataset':'bones','source_family_id':row['source_family_id'],'source_metadata':row,'source_semantics':semantics,'status':'KINEMATIC_CANDIDATE','source_sha256':row['sealed_native_sha256'],'frames':len(q),'contract_status':c['status'],'output_paths':paths,'output_hashes':hashes,'metrics_path':str((out/'motion/solver.json').relative_to(ROOT)),'rejections':[],'review_flags':[],'gmr_role':'official_g1_main_pool'},{}


def prepare(task,out):
 row=task['row'];source=row['dataset'];task=copy.deepcopy(task);folder=out/'native_mapping';folder.mkdir()
 if source=='humanml3d':
  from .native_rest_mapping import smplh_mapping
  path=Path(row['sealed_native_path']);cfg,audit=smplh_mapping(path,mirror=row['mirror_flag'])
 else:
  import zstandard as zstd
  from .hiphi_rest_mapping import mapping
  pp=row['package'];cp=ROOT/pp['compressed_path']
  if sha(cp)!=pp['compressed_sha256']:raise ValueError('STAGED_NATIVE_INPUT_CHANGED')
  path=folder/'motion.bvh'
  with cp.open('rb') as src,path.open('wb') as dst:zstd.ZstdDecompressor().copy_stream(src,dst)
  cfg,audit=mapping(path,row['height_m'])
  task['package']=pp;task['package_path']=str((folder/'package.json').relative_to(ROOT));write_json(ROOT/task['package_path'],pp)
 if sha(path)!=row['sealed_native_sha256']:raise ValueError('SEALED_NATIVE_INPUT_CHANGED')
 config=folder/'mapping.json';auditpath=folder/'mapping_audit.json';write_json(config,cfg);write_json(auditpath,audit)
 task['mapping']={'config_path':str(config.relative_to(ROOT)),'config_sha256':sha(config),'audit_path':str(auditpath.relative_to(ROOT)),'audit_sha256':sha(auditpath),'native_sha256':row['sealed_native_sha256'],'native_path':str(path.relative_to(ROOT)) if source=='hiphi' else str(path),'height_discrepancy_above_10_percent':audit['source_provenance'].get('height_discrepancy_above_10_percent',False)}
 task.setdefault('package_path',str(config.relative_to(ROOT)))
 return task


def execute(task,out,asset,expected_code):
 start=time.monotonic();row=task['row'];authorize(row['dataset']);prior=load(BASE/'prior_pilot_outputs.json').get(row['sample_id'])
 if prior:
  entry=copy.deepcopy(prior['entry'])
  for key in ('source_paths','native_interval','mirror_flag','captions','source_family_id'):
   if entry['source_metadata'].get(key)!=row.get(key):raise ValueError('REUSE_SOURCE_METADATA_MISMATCH:'+key)
  if entry['source_sha256']!=row['sealed_native_sha256']:raise ValueError('REUSE_NATIVE_MISMATCH')
  for path,bound in entry['output_hashes'].items():
   if sha(ROOT/path)!=bound:raise ValueError('REUSE_OUTPUT_MISMATCH')
  if sha(ROOT/entry['mapping_config'])!=entry['mapping_sha256']:raise ValueError('REUSE_MAPPING_MISMATCH')
  entry.update(source_metadata=row,reused=True,reuse_namespace=prior['namespace']);execution={'reused':True,'new_solver_attempts':0}
 elif row['dataset']=='bones':entry,execution=bones(task,out)
 else:
  staged=prepare(task,out);compile_authorized_process()
  entry,execution=execute_original(staged,out,asset,expected_code)
  # Only the decompressed scratch copy is removed; sealed compressed native is retained.
  if row['dataset']=='hiphi':(out/'native_mapping/motion.bvh').unlink()
  entry.update(body_only_validated=row['dataset']=='hiphi',reused=False)
 if entry.get('status')=='ERROR':return entry,execution
 if not entry.get('output_paths'):return entry,execution
 metrics=load(ROOT/entry['metrics_path']);contract=load(ROOT/entry['output_paths']['contract_path'])
 if contract['asset_hash']!=asset['report']['source']['asset_hash']:raise ValueError('REUSE_TARGET_ASSET_MISMATCH')
 decision=decide(entry,metrics,contract,prior=prior)
 dp=out/'kinematic_decision.json';write_json(dp,decision)
 entry['output_hashes'][str(dp.relative_to(ROOT))]=sha(dp)
 entry.update(decision=decision['decision'],decision_reasons=decision['reasons'],diagnostic_tags=decision['diagnostic_tags'],decision_path=str(dp.relative_to(ROOT)),dynamic_validation='NOT_RUN',execution_scope='kinematic_training')
 entry['status']={'ACCEPTED_KINEMATIC':'KINEMATIC_CANDIDATE','REJECTED':'REJECTED','REVIEW_PENDING':'REVIEW_REQUIRED'}[decision['decision']]
 execution['total_clip_sec_including_mapping']=time.monotonic()-start
 return entry,execution
