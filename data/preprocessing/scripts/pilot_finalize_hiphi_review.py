"""Bind completed agent review notes to the unchanged pilot evaluator.

Requires real, complete review notes. Does not infer visual observations or
promote unreviewed clips, alter classifications, or write production gates.
"""
import json
import math
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha
from g1_pipeline.calibration_protocol import PATH
from g1_pipeline.source_smoke_evaluation import evaluate


def run():
 ns='pilot_hiphi100_native_v1_eb91dfef770f'
 base=ROOT/'work'/ns;records=read_jsonl(base/'results.jsonl')
 notespath=ROOT/'audit'/f'{ns}_visual_notes.json'
 notes=json.loads(notespath.read_text())['results']
 if len(records)!=100 or len(notes)!=100 or len({r['sample_id'] for r in notes})!=100:
  raise ValueError('COMPLETE_100_REAL_REVIEW_NOTES_REQUIRED')
 notes={n['sample_id']:n for n in notes}
 if set(notes)!={r['sample_id'] for r in records}:raise ValueError('REVIEW_PANEL_MISMATCH')
 videospath=ROOT/'videos'/ns/'manifest.json'
 videos={r['sample_id']:r for r in json.loads(videospath.read_text())['videos']}
 decisions=[];extra={str(notespath.relative_to(ROOT)):sha(notespath)}
 gaps={'hiphi:Self_motion-vault_0011','hiphi:Self_motion-scramble_0002'}
 for row in records:
  sid=row['sample_id'];note=notes[sid];v=videos[sid]
  if sha(ROOT/note['sheet_path'])!=note['sheet_sha256'] or note['sheet_sha256']!=v['sheet_sha256']:
   raise ValueError('REVIEW_SHEET_CHANGED:'+sid)
  if not note.get('observation') or not note.get('review_method'):raise ValueError('REVIEW_OBSERVATION_REQUIRED')
  disposition=note['proposed_disposition']
  if disposition not in ('REJECT','RETAIN_KINEMATIC_ONLY'):raise ValueError('INVALID_REVIEW_DISPOSITION')
  if sid in gaps:
   assert row['source_semantics']['scene_class']=='flat_ground_core_candidate'
   assert row['source_semantics']['support_evidence']==[]
   if disposition!='REJECT':raise ValueError('UNRESOLVED_SUPPORT_COVERAGE_MUST_REMAIN_REJECTED')
  m=json.loads((ROOT/row['metrics_path']).read_text())
  ground=(m['ground_correction'] or {}).get('lift_max_m',0)
  rate=(m['rate_projection'] or {}).get('correction_max_rad',0)
  spectrum=math.degrees(m.get('native_spectrum',{}).get('high_band_displacement_rms_max',0))
  rationale=(note['observation']+f" Full-series contract={row['contract_status']}; ground lift max={ground:.6f}m; rate correction max={rate:.6f}rad; "
             f"native high-band angular displacement RMS max={spectrum:.6f}deg. "
             "Existing numerical, ground, collision, hinge, skating and round-trip gates remain mandatory. "
             "No filtering, relabeling or shared solver tuning. Kinematic pilot review only; no dynamic acceptance, human signature or exhaustive visual inspection.")
  decisions.append({'sample_id':sid,'disposition':disposition,'rationale':rationale,'visual_evidence':[note['sheet_path']]})
  for path,bound in note.get('additional_evidence',{}).items():
   if sha(ROOT/path)!=bound:raise ValueError('ADDITIONAL_REVIEW_EVIDENCE_CHANGED')
   extra[path]=bound
 # Construct the exact existing evaluator's evidence set without a stale review.
 manifest=ROOT/'manifests/hiphi_smoke100.jsonl'
 paths=[manifest,base/'results.jsonl',base/'run.json',PATH,ROOT/'audit/hiphi_native_rest_mapping_v1.json',ROOT/'audit/hiphi_package_evidence_v1.json',videospath]
 bindings={str(p.relative_to(ROOT)):sha(p) for p in paths}
 for v in videos.values():
  for key,hashkey in [('path','sha256'),('sheet_path','sheet_sha256')]:
   if sha(ROOT/v[key])!=v[hashkey]:raise ValueError('VIDEO_CHANGED')
   bindings[v[key]]=v[hashkey]
  for path,bound in v['input_hashes'].items():
   if sha(ROOT/path)!=bound:raise ValueError('VIDEO_SOURCE_CHANGED')
 bindings.update(extra)
 result={'execution_scope':'isolated_pilot','training_release':False,'policy_sha256':sha(PATH),
         'review_method':'Agent sampled-frame inspection with full-series numerical metrics; all full 25FPS videos exist. No human sign-off or exhaustive frame-by-frame review claimed.',
         'bindings':bindings,'decisions':decisions,
         'systematic_checks':{'left_right':'PASS','scale':'PASS','time':'PASS','joint_order':'PASS','native_geometry':'PASS','support_scene':'FAIL'},
         'systematic_check_rationale':'Fixed source-specific native rest/FK mapping and per-file BVH timing, complete exporter/representation/round-trip checks and sampled asymmetric motion provide axis, units, scale, joint-order and time evidence. Original Self_motion-vault_0011 and Self_motion-scramble_0002 remain flat_ground_core_candidate with support_evidence=[] despite native hand-support content; this unresolved coverage gap is conservatively rejected and support_scene remains FAIL. No source relabeling, denominator change or blanket source PASS. Individual hard correction/physics outliers remain rejected.'}
 write_json(ROOT/f'audit/{ns}_review_decisions.json',result)
 report=evaluate(ns,'hiphi',pilot=True)
 print({k:report[k] for k in ['status','attempted','kinematic_retained','failure_reasons','review_pending_ids']})
if __name__=='__main__':run()
