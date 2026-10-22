"""Seal the existing candidate inventory and materialize only Body-only BVH members."""
from motion_mamba2.paths import artifact_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv,hashlib,json,tarfile,io,collections,concurrent.futures,shutil
from pathlib import Path,PurePosixPath
import zstandard as zstd
from g1_pipeline.common import ROOT,HML,HIPHI,BONES,sha,digest,write_json,write_jsonl,atomic_bytes,read_jsonl,now
BASE=ROOT/'releases/kinematic_training_20260917_v1'
REQUEST=Path(artifact_path('9_17-_md_65267f5053'))

def stage_archive(args):
 archive,rows=args;dst=BASE/'inputs/hiphi';dst.mkdir(parents=True,exist_ok=True)
 index=dst/(archive+'.index.json')
 if index.exists():
  old=json.loads(index.read_text());p=HIPHI/'data'/archive
  if old['archive_stat']!={'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}:raise ValueError('ARCHIVE_CHANGED')
  for r in old['results']:
   if r.get('compressed_path') and sha(ROOT/r['compressed_path'])!=r['compressed_sha256']:raise ValueError('STAGED_INPUT_CHANGED')
  return old
 selected={r['source_motion_id']:r for r in rows};found={k:{'sample_id':v['sample_id'],'members':[]} for k,v in selected.items()}
 p=HIPHI/'data'/archive;initial={'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns};hasher=hashlib.sha256()
 class HashedReader(io.RawIOBase):
  def __init__(self,raw):self.raw=raw
  def readable(self):return True
  def read(self,n=-1):
   b=self.raw.read(n);hasher.update(b);return b
  def readinto(self,b):
   data=self.read(len(b));b[:len(data)]=data;return len(data)
 with p.open('rb') as raw:
  tracked=HashedReader(raw)
  with zstd.ZstdDecompressor().stream_reader(tracked,closefd=False) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
   for member in tar:
    parts=PurePosixPath(member.name).parts
    if len(parts)<2:continue
    mid=parts[-2]
    if mid not in selected:
     if 'object_tracks' in parts:
      for part in parts:
       if part in found:found[part]['object_tracks']=True
     continue
    r=found[mid];r['members'].append(member.name)
    if not member.isfile():continue
    if parts[-1]=='metadata.json':
     data=tar.extractfile(member).read();r['metadata']=json.loads(data);r['metadata_sha256']=hashlib.sha256(data).hexdigest()
    elif parts[-1]=='motion_actor.bvh':
     data=tar.extractfile(member).read();r['native_sha256']=hashlib.sha256(data).hexdigest();r['native_bytes']=len(data)
     out=dst/(mid+'.bvh.zst');atomic_bytes(out,zstd.ZstdCompressor(level=3).compress(data))
     r.update(compressed_path=str(out.relative_to(ROOT)),compressed_sha256=sha(out))
   # Consume trailing compressed bytes so archive hash is a complete file hash.
   while stream.read(8<<20):pass
  while tracked.read(8<<20):pass
 if initial!={'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}:raise ValueError('ARCHIVE_CHANGED_DURING_STAGING')
 for mid,r in found.items():
  row=selected[mid];m=r.get('metadata',{});reasons=[]
  if m.get('motion_id')!=mid or m.get('dataset')!='HiPHI':reasons.append('PACKAGE_IDENTITY_MISMATCH')
  if m.get('is_hoi') is not False or m.get('objects')!=[] or r.get('object_tracks'):reasons.append('BODY_ONLY_NOT_PROVEN')
  if m.get('mirrored') is not row['mirror_flag']:reasons.append('MIRROR_METADATA_MISMATCH')
  if m.get('frame_count')!=row['original_frames']:reasons.append('FRAME_COUNT_MISMATCH')
  if m.get('actor_id')!=row['actor_id_namespaced'].split(':',1)[1]:reasons.append('ACTOR_MISMATCH')
  if m.get('frame_lu')!=row['category']+'-'+row['lu'] or m.get('text_annotation') not in row['captions']:reasons.append('SEMANTIC_METADATA_MISMATCH')
  if abs(float(m.get('actor_metadata',{}).get('height_cm',-1))/100-row['height_m'])>1e-8:reasons.append('HEIGHT_MISMATCH')
  if 'compressed_path' not in r:reasons.append('MISSING_NATIVE_BVH')
  r.update(status='INPUT_METADATA_PASS' if not reasons else 'UNSUPPORTED',failures=reasons,bvh_member_sha256=r.get('native_sha256'))
 result={'archive':archive,'archive_path':str(p),'archive_sha256':hasher.hexdigest(),'archive_stat':initial,'results':list(found.values())}
 write_json(index,result);print(json.dumps({'phase':'archive_sealed','archive':archive,'body_only_candidates':len(found),'issues':sum(bool(r['failures']) for r in found.values())}),flush=True)
 return result

def run():
 BASE.mkdir(parents=True,exist_ok=True)
 if (BASE/'input_release.json').exists():print('Existing sealed input release; no re-staging');return
 write_json(BASE/'preparation_status.json',{'time':now(),'status':'RUNNING','pid':os.getpid(),'phase':'inventory_and_native_hashes'})
 rows=read_jsonl(ROOT/'manifests/full_candidates.v2.jsonl')
 assert len(rows)==len({r['sample_id'] for r in rows})
 heldout={r['source_family_id'] for r in read_jsonl(ROOT/'manifests/bones_pairs_accept.jsonl')}
 prior={};reused={};bad_families=set()
 for short in ['hml','hiphi']:
  ns=json.loads((ROOT/f'audit/current_pilot_{short}100_run.json').read_text())['namespace']
  decisions={r['sample_id']:r for r in json.loads((ROOT/f'audit/{ns}_review_decisions.json').read_text())['decisions']}
  for r in read_jsonl(ROOT/'work'/ns/'results.jsonl'):
   reused[r['sample_id']]={'entry':r,'decision':decisions[r['sample_id']],'namespace':ns,'run_sha256':sha(ROOT/'work'/ns/'run.json')}
   if short=='hml' and r['source_metadata']['category']=='DFaust_67':bad_families.add(r['source_family_id'])
 write_json(BASE/'prior_pilot_outputs.json',reused)
 meta={r['filename']:r['content_technical_description'] for r in csv.DictReader((BONES/'metadata/seed_metadata_v004.csv').open())}
 candidates=[];isolated=[];native_paths=set();archives=collections.defaultdict(list)
 for r in rows:
  r=dict(r);reason=r.get('reason') if r['eligibility']!='candidate' else None
  if r['source_family_id'] in bad_families:reason='CONFIRMED_DFAUST_NATIVE_FLOOR_ORIGIN_FAMILY'
  r['exposure_history']=['historical_pilot'] if r['sample_id'] in reused else []
  r['training_role']='held_out_calibration' if r['source_family_id'] in heldout else r['project_split']
  if r['dataset']=='bones':r['source_description']=meta[r['source_motion_id']]
  if reason:
   isolated.append({'row':r,'decision':'UNSUPPORTED','reasons':[reason],'attempted':False});continue
  if r['dataset']=='hiphi':archives[r['archive']].append(r)
  else:
   p=(BONES/r['source_paths']['g1']) if r['dataset']=='bones' else HML/r['source_paths']['native']
   if not p.is_file():isolated.append({'row':r,'decision':'UNSUPPORTED','reasons':['MISSING_NATIVE_INPUT'],'attempted':False});continue
   r['sealed_native_path']=str(p);native_paths.add(str(p))
  candidates.append(r)
 cache=BASE/'inputs/native_hashes.json'
 if cache.exists():hashes=json.loads(cache.read_text())
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:hashes=dict(zip(sorted(native_paths),pool.map(sha,sorted(native_paths))))
  write_json(cache,hashes)
 print(json.dumps({'phase':'native_hashes_sealed','files':len(hashes),'candidates':len(candidates),'isolated':len(isolated)}),flush=True)
 write_json(BASE/'preparation_status.json',{'time':now(),'status':'RUNNING','pid':os.getpid(),'phase':'body_only_archive_staging','archives':len(archives)})
 # Two I/O readers, no solver/GPU use; only compressed Body-only members retained.
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:packages=list(pool.map(stage_archive,sorted(archives.items())))
 hiphi={r['sample_id']:r for p in packages for r in p['results']}
 valid=[]
 for r in candidates:
  if r['dataset']=='hiphi':
   record=hiphi[r['sample_id']]
   if record['status']!='INPUT_METADATA_PASS':isolated.append({'row':r,'decision':'UNSUPPORTED','reasons':record['failures'],'attempted':False});continue
   r['sealed_native_sha256']=record['native_sha256'];r['package']=record
  else:r['sealed_native_sha256']=hashes[r['sealed_native_path']]
  valid.append(r)
 write_jsonl(BASE/'inventory.jsonl',rows);write_jsonl(BASE/'isolated.jsonl',isolated)
 # Fixed source/category/duration strata, then long-first within later chunks.
 groups=collections.defaultdict(list)
 for r in valid:
  duration=r.get('metadata_duration_sec') or (r['native_end_sec']-r['native_start_sec'])
  r['estimated_target_frames']=max(1,round(duration*25))
  groups[(r['dataset'],r['category'],0 if duration<5 else 1 if duration<30 else 2)].append(r)
 for g in groups.values():g.sort(key=lambda r:digest([20260917,r['sample_id']]))
 first=[]
 while len(first)<min(1000,len(valid)):
  for key in sorted(groups):
   if groups[key] and len(first)<1000:first.append(groups[key].pop())
 selected={r['sample_id'] for r in first}
 remaining=sorted((r for r in valid if r['sample_id'] not in selected),key=lambda r:(-r['estimated_target_frames'],digest(r['sample_id'])))
 order=first+remaining;write_jsonl(BASE/'candidates.jsonl',order)
 # Frozen panels use existing per-clip scheduler, no new queue service.
 panels=[]
 for start in range(0,len(order),1000):
  chunk=order[start:start+1000];chunk=sorted(chunk,key=lambda r:-r['estimated_target_frames'])
  p=BASE/'panels'/f'batch_{start//1000:04d}.json'
  write_json(p,{'scope':'kinematic_training','release':str(BASE.relative_to(ROOT)),'tasks':[{'source':r['dataset'],'row':r} for r in chunk]})
  panels.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'clips':len(chunk),'frames_estimate':sum(r['estimated_target_frames'] for r in chunk)})
 bindings={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'inventory.jsonl',BASE/'isolated.jsonl',BASE/'candidates.jsonl',BASE/'prior_pilot_outputs.json',cache]}
 bindings.update({str(ROOT/'manifests/full_candidates.v2.jsonl'):sha(ROOT/'manifests/full_candidates.v2.jsonl'),str(REQUEST):sha(REQUEST)})
 for a in archives:bindings[str((BASE/'inputs/hiphi'/(a+'.index.json')).relative_to(ROOT))]=sha(BASE/'inputs/hiphi'/(a+'.index.json'))
 for p in [BONES/'metadata/seed_metadata_v004.csv',HIPHI/'metadata/hiphi_metadata.csv',HIPHI/'metadata/hiphi_actor_metadata.csv']:bindings[str(p)]=sha(p)
 report={'time':now(),'scope':'kinematic_training','status':'INPUTS_SEALED','request_path':str(REQUEST),'request_sha256':sha(REQUEST),'inventory':len(rows),'supported_candidates':len(valid),'isolated':len(isolated),'source_inventory':dict(collections.Counter(r['dataset'] for r in rows)),'source_candidates':dict(collections.Counter(r['dataset'] for r in valid)),'source_isolated':dict(collections.Counter(r['row']['dataset'] for r in isolated)),'input_bindings':bindings,'panels':panels,'heldout_families':sorted(heldout),'floor_origin_quarantine_families':sorted(bad_families),'split_policy':'original project/official splits retained; heldout families excluded from training and stats; mirror/take/crop family constraints retained','first1000':'source/category/duration stratified, seed20260917; operational window, not a yield gate','archive_bindings':{p['archive_path']:p['archive_sha256'] for p in packages},'disk_free_bytes':shutil.disk_usage(BASE).free}
 write_json(BASE/'input_release.json',report);write_json(BASE/'preparation_status.json',{'time':now(),'status':'COMPLETE','inventory':len(rows),'supported_candidates':len(valid),'isolated':len(isolated)})
 print(json.dumps({k:report[k] for k in ['status','inventory','supported_candidates','isolated','source_candidates','disk_free_bytes']}),flush=True)
if __name__=='__main__':run()
