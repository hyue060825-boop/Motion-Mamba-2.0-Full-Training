"""Explicit 2026-09-17 offline-only policy, independent of historical gates."""
import json
from .common import ROOT,sha
BASE=ROOT/'releases/kinematic_training_20260917_v1'

def load(path):return json.loads(path.read_text())

def authorize(source=None):
 p=load(BASE/'execution_policy.json')
 if p['scope']!='kinematic_training' or p['frozen_solver']!='v25' or p['aggregate_retention_min'] is not None or p['dynamic_required']:
  raise ValueError('INVALID_KINEMATIC_AUTHORIZATION')
 if sha(p['request_path'])!=p['request_sha256']:raise ValueError('KINEMATIC_REQUEST_CHANGED')
 if source is not None and source not in ('bones','humanml3d','hiphi'):raise ValueError('UNKNOWN_SOURCE')
 return p

def decide(entry,metrics,contract,*,prior=None):
 """Hard failures precede diagnostic reinterpretation; uncertain cases stay pending."""
 policy=authorize(entry['dataset']);scene=entry['source_semantics']['scene_class']
 original=list(entry.get('rejections',[]));diagnostic=[];hard=[];pending=[]
 soft={'RELATIVE_ACTION_CLASS_UNSUPPORTED','RELATIVE_CATEGORY_UNSUPPORTED'}
 conditional={'GROUND_PENETRATION','GROUND','FOOT_SKATING','SKATING','EXTERNAL_OR_UNRESOLVED_SUPPORT_SCENE'}
 for reason in original:
  if reason in soft:diagnostic.append(reason)
  elif reason in conditional and scene!='flat_ground_core_candidate':diagnostic.append(reason)
  else:hard.append(reason)
 if contract.get('status')!='PASS':hard.append('NUMERICAL_CONTRACT_FAILURE')
 if not contract.get('finite'):hard.append('NONFINITE_CONTRACT')
 if contract.get('hinge_limit_excess_max_rad',1)>1e-4:hard.append('HINGE_LIMIT')
 rate=(metrics.get('rate_projection') or {}).get('correction_max_rad',0)
 if rate>policy['maximum_hinge_correction_rad']:hard.append('RATE_CORRECTION_EXCEEDS_CONTENT_CHANGE_BUDGET')
 geom=metrics.get('geometry',{})
 if not geom:pending.append('MISSING_FULL_FRAME_GEOMETRY')
 else:
  if geom['raw_self_collision_depth_over_1cm_frame_fraction']>.001 or geom['raw_self_collision_max_depth_m']>.03:hard.append('SELF_COLLISION')
  if scene=='flat_ground_core_candidate':
   if geom['ground_penetration_p95_m']>.02 or geom['ground_penetration_max_m']>.05:hard.append('GROUND')
   if any(v is not None and v>.2 for v in contract['skating_p95_m_s'].values()):hard.append('SKATING')
  elif scene in ('unclassified','object_interaction_review','support_ambiguous_review'):pending.append('UNRESOLVED_APPLICABLE_SCENE')
  else:diagnostic.append('GROUND_CONTACT_SCENE_DIAGNOSTIC')
 flags=list(entry.get('review_flags',[]));diagnostic+=flags
 bound_review=prior and prior['decision']['disposition']=='RETAIN_KINEMATIC_ONLY'
 if prior:
  # Never turn old physics/numerical hard rejections into automatic acceptance.
  hard.extend(r for r in prior['entry'].get('rejections',[]) if r not in soft|{'EXTERNAL_OR_UNRESOLVED_SUPPORT_SCENE'})
  if prior['decision']['disposition']=='REJECT':pending.append('PRIOR_REVIEW_REQUIRES_INDEPENDENT_RESOLUTION')
 if entry['dataset']!='bones':
  if not entry.get('native_provenance_validated'):hard.append('NATIVE_PROVENANCE_MISSING')
  if not metrics.get('human_target_metrics'):pending.append('MISSING_SOURCE_TARGET_FIDELITY_EVIDENCE')
  if 'CORRECTION_CONTENT_REVIEW_REQUIRED' in flags and not bound_review:pending.append('CORRECTION_CONTENT_REVIEW_REQUIRED')
  if 'NATIVE_SUPPORT_POSTURE_REVIEW_REQUIRED' in flags and not bound_review:pending.append('SOURCE_SUPPORT_POSTURE_UNRESOLVED')
  if entry['dataset']=='humanml3d' and entry['source_metadata'].get('category')=='DFaust_67':pending.append('DFAUST_NATIVE_FLOOR_ORIGIN_NOT_VALIDATED')
 # Diagnostic frequency/relative labels alone do not block kinematic data.
 decision='REJECTED' if hard else 'REVIEW_PENDING' if pending else 'ACCEPTED_KINEMATIC'
 return {'decision':decision,'reasons':sorted(set(hard or pending)),'diagnostic_tags':sorted(set(diagnostic)),
         'scene_class':scene,'dynamic_validation':'NOT_RUN','policy_sha256':sha(BASE/'execution_policy.json'),
         'prior_decision':prior['decision'] if prior else None,'prior_solver_status':prior['entry']['status'] if prior else None,
         'fidelity_evidence':'official G1 source + full-series contract' if entry['dataset']=='bones' else 'native shape/rest/FK, per-file physical time, scaled human targets, initial/raw/final robot arrays and full-series correction/QA metrics',
         'review_basis':'bound historical review reused' if bound_review else 'full-frame automated QA; release-level stratified semantic audit still required'}
