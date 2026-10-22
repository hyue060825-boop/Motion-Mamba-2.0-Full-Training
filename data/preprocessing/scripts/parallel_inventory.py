"""Read current release candidates for ETA denominators, without production admission."""
import json
from collections import defaultdict,Counter
from g1_pipeline.common import ROOT,sha,write_json,now


def run():
 path=ROOT/'manifests/full_candidates.v2.jsonl'
 groups=defaultdict(lambda:{'records':0,'candidate_records':0,'candidate_seconds_metadata':0.,'unknown_candidate_duration':0,
                           'train_candidate_records':0,'train_candidate_seconds_metadata':0.,'unmirrored_candidate_records':0,
                           'unmirrored_candidate_seconds_metadata':0.,'eligibility':Counter()})
 for line in path.open():
  r=json.loads(line);g=groups[r['dataset']];g['records']+=1;g['eligibility'][r['eligibility']]+=1
  if r['eligibility']!='candidate':continue
  g['candidate_records']+=1
  duration=r['native_end_sec']-r['native_start_sec'] if r.get('native_end_sec') is not None else r.get('metadata_duration_sec')
  if duration is None:g['unknown_candidate_duration']+=1
  else:g['candidate_seconds_metadata']+=duration
  if r['project_split']=='train':
   g['train_candidate_records']+=1;g['train_candidate_seconds_metadata']+=duration or 0
  if not r['mirror_flag']:
   g['unmirrored_candidate_records']+=1;g['unmirrored_candidate_seconds_metadata']+=duration or 0
 for source,g in groups.items():
  g['estimated_candidate_25hz_frames']=g['candidate_seconds_metadata']*25
  g['known_candidate_hours']=g['candidate_seconds_metadata']/3600
  g['duration_authority']='native interval metadata, per-file validation required' if source!='hiphi' else 'HiPHI metadata duration; full library BVH Frame Time/package validation still absent'
  g['production_admitted_records']=0
 old=json.loads((ROOT/'audit/full_resource_budget_v1.json').read_text())
 report={'time':now(),'status':'CURRENT_METADATA_INVENTORY_ONLY','manifest_path':str(path.relative_to(ROOT)),
         'manifest_sha256':sha(path),'sources':dict(groups),'production_scope_finalized':False,
         'historical_371_8_hours_source':'HiPHI candidate metadata records including mirror augmentation',
         'historical_manifest_matches_current':old['manifest_sha256']==sha(path),
         'limitations':['Metadata candidates include unsupported scene classes not admitted by a valid new production release.',
                       'Candidate frames are duration*25 estimates, not materialized full-library resampling counts.',
                       'Mirrored records count output work, not unique captured motion hours.',
                       'BONES official reference export has a different execution path; HML/HiPHI retarget benchmark cannot establish BONES throughput.',
                       'Native archive staging, model cache, video selection, exception review and production dynamic QA are separate costs.']}
 write_json(ROOT/'audit/parallel_eta_inventory.json',report)
 print({s:{k:g[k] for k in ['candidate_records','known_candidate_hours','estimated_candidate_25hz_frames','unknown_candidate_duration']} for s,g in groups.items()})
if __name__=='__main__':run()
