"""Localize historical QP rounding using unchanged frozen numeric contracts.

CSV/features/time/contracts/flags/decisions and pre-postprocessing arrays must
remain exact. Every allowed scalar difference is independently checked. No
solver, output, policy or hard QA threshold is modified by this checker.
"""
import json
import math
import numpy as np
import yaml
from g1_pipeline.common import ROOT,sha


def differences(a,b,path=()):
 if isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):return [(path,a,b)]
  return [d for k in a for d in differences(a[k],b[k],path+(k,))]
 if isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):return [(path,a,b)]
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+(i,))]
 return [] if a==b else [(path,a,b)]


def at(value,path):
 for key in path:value=value[key]
 return value


def ground_derived(report):
 if report is None:return None
 lift=np.asarray(report['lift_m'])
 return {'lift_max_m':float(lift.max()),'lift_mean_m':float(lift.mean()),
         'added_vertical_speed_max_m_s':float(np.max(abs(np.diff(lift)/.04))),
         'added_vertical_acceleration_max_m_s2':float(np.max(abs(np.diff(lift,n=2)/.04**2)))}


def check(ra,ca,issues):
 if not issues or not set(issues)<= {'qpos.npy','solver.json'}:return None
 diagnostic=ROOT/'audit/parallel_projection_roundoff_diagnostic.json'
 if not diagnostic.exists():return None
 diagnosis=json.loads(diagnostic.read_text())
 if not diagnosis['max_pairwise_output_difference_rad']>0:return None
 if sha(ROOT/'g1_pipeline/bounded_hinge_trajectory.py')!=diagnosis['projection_code_sha256']:return None
 a=np.load(ra/'qpos.npy');b=np.load(ca/'qpos.npy')
 raw_a=np.load(ra/'unprocessed_qpos.npy');raw_b=np.load(ca/'unprocessed_qpos.npy')
 if not np.array_equal(raw_a,raw_b) or not np.array_equal(a[:,[0,1,3,4,5,6]],b[:,[0,1,3,4,5,6]]):return None
 ma=json.loads((ra/'solver.json').read_text());mb=json.loads((ca/'solver.json').read_text())
 rate_a=ma['rate_projection'];rate_b=mb['rate_projection']
 rate_stats={'correction_mae_rad','correction_max_rad'}
 if rate_a is None or rate_b is None:
  if rate_a!=rate_b:return None
 elif {k:v for k,v in rate_a.items() if k not in rate_stats}!={k:v for k,v in rate_b.items() if k not in rate_stats}:return None
 changed=set(np.flatnonzero(np.any(a[:,7:]!=b[:,7:],axis=0)).tolist())
 if not changed<=set((rate_a or {}).get('changed_joint_indices',[])):return None
 gates=ROOT/'configs/gates.v2.yaml';numeric=yaml.safe_load(gates.read_text())['numeric']
 maxrad=float(np.max(np.abs(a[:,7:]-b[:,7:])));maxdeg=math.degrees(maxrad)
 maxroot=float(np.max(abs(a[:,:3]-b[:,:3])))
 if maxdeg>numeric['hinge_roundtrip_max_deg'] or maxroot>numeric['root_translation_roundtrip_max_m']:return None
 ground_a=ma['ground_correction'];ground_b=mb['ground_correction'];ground_diag=None
 if ground_a!=ground_b or not np.array_equal(a[:,2],b[:,2]):
  ground_diag=ROOT/'audit/parallel_ground_roundoff_diagnostic.json'
  if not ground_diag.exists():return None
  gd=json.loads(ground_diag.read_text())
  if sha(ROOT/'g1_pipeline/ground_clearance.py')!=gd['ground_code_sha256'] or not any(r['serial_repeat_max_abs_m']>0 for r in gd['results']):return None
  if ground_a is None or ground_b is None:return None
  allowed={'lift_m',*ground_derived(ground_a)}
  if {k:v for k,v in ground_a.items() if k not in allowed}!={k:v for k,v in ground_b.items() if k not in allowed}:return None
  if not np.array_equal(a[:,2],raw_a[:,2]+np.asarray(ground_a['lift_m'])) or not np.array_equal(b[:,2],raw_b[:,2]+np.asarray(ground_b['lift_m'])):return None
  if np.max(abs(np.asarray(ground_a['lift_m'])-ground_b['lift_m']))>numeric['root_translation_roundtrip_max_m']:return None
 # Byte-identical consumer artifacts impose a stronger restriction than the
 # existing position/angle round-trip bounds alone.
 for name in ['motion.csv','motion.json','features.npy','time.json','contract.json','human_targets.npz']:
  if name.endswith('.npz'):
   with np.load(ra/name) as x,np.load(ca/name) as y:
    if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):return None
  elif sha(ra/name)!=sha(ca/name):return None
 ds=differences(ma,mb);verified={}
 from g1_pipeline.robot import compile_target
 from g1_pipeline.bones_continuity_qa import metrics as continuity
 from g1_pipeline.bones_relative_motion_v2 import series
 from g1_pipeline.bones_pair_qa import replay
 m,_=compile_target()
 with np.load(ra/'human_targets.npz') as targets:t=targets['timestamps'].copy()
 cont_a=continuity(m,a,t);cont_b=continuity(m,b,t)
 ser_a=series(m,a,t);ser_b=series(m,b,t)
 geometry_a=geometry_b=None
 ga=ground_derived(ground_a);gb=ground_derived(ground_b)
 correction_a=abs(a[:,7:]-raw_a[:,7:]);correction_b=abs(b[:,7:]-raw_b[:,7:])
 rate_derived_a={'correction_mae_rad':float(correction_a.mean()),'correction_max_rad':float(correction_a.max())}
 rate_derived_b={'correction_mae_rad':float(correction_b.mean()),'correction_max_rad':float(correction_b.max())}
 policy=json.loads((ROOT/'configs/calibration_protocol.v25.json').read_text());guard=policy['human_target_correction_review']
 for path,x,y in ds:
  if not isinstance(x,float) or not isinstance(y,float) or not np.isfinite([x,y]).all():return None
  rule=None
  if path[0]=='continuity':
   if at(cont_a,path[1:])!=x or at(cont_b,path[1:])!=y:return None
   rule='Exactly reproduced from each saved float64 trajectory by unchanged continuity QA; no metric tolerance'
  elif path[0]=='relative_motion' and path[2] in ('joint_speed_rad_s','joint_acceleration_rad_s2') and path[3]=='per_coordinate_max':
   key,j=path[2],path[4]
   if float(ser_a[key].max(axis=0)[j])!=x or float(ser_b[key].max(axis=0)[j])!=y:return None
   rule='Exactly reproduced from unchanged relative series; all threshold ratios/flag indices and decisions still exact'
  elif path[0]=='ground_correction':
   if path[1]=='lift_m':
    # Complete lifts were validated above against root Z and original numeric tolerance.
    rule='Existing frozen root-position tolerance; original clearance lower bounds/settings exact; saved root Z exactly equals original Z plus this full lift array'
   elif path[1] in ga:
    if ga[path[1]]!=x or gb[path[1]]!=y:return None
    rule='Exactly recomputed from the complete saved lift array using the unchanged ground report formula'
  elif path[0]=='rate_projection' and path[1] in rate_stats:
   if rate_derived_a[path[1]]!=x or rate_derived_b[path[1]]!=y:return None
   rule='Exactly recomputed correction mean/max from saved final and pre-correction hinges; unchanged solver controls and changed-joint set'
  elif path[0]=='geometry':
   if geometry_a is None:geometry_a=replay(m,a)[2];geometry_b=replay(m,b)[2]
   if at(geometry_a,path[1:])!=x or at(geometry_b,path[1:])!=y:return None
   rule='Exactly reproduced by unchanged complete geometry replay; all hard rejection reasons and decisions still exact'
  elif path[0] in ('human_target_metrics','human_target_residual_change'):
   metric=path[-1]
   if metric.startswith('position_') and metric.endswith('_m'):limit=numeric['fk_position_max_m']
   elif metric.startswith('orientation_') and metric.endswith('_deg'):limit=numeric['fk_orientation_max_deg']
   else:return None
   if abs(x-y)>limit or np.float32(x)!=np.float32(y):return None
   if path[0]=='human_target_residual_change':
    threshold=guard['position_increase_above_m' if metric.startswith('position_') else 'orientation_increase_above_deg']
    if (x>threshold)!=(y>threshold):return None
   rule='Existing frozen FK-unit tolerance and equal at already-exported float32 residual precision; target arrays exact and correction-review threshold sides unchanged'
  if rule is None:return None
  group='/'.join(map(str,path[:-1] if isinstance(path[-1],int) else path))
  item=verified.setdefault(group,{'path':group,'count':0,'max_abs':-1,'verification':rule})
  item['count']+=1
  if abs(x-y)>item['max_abs']:item.update(reference=x,candidate=y,max_abs=abs(x-y),maximum_at='/'.join(map(str,path)))
 return {'status':'PASS_DIAGNOSED_HISTORICAL_QP_ROUNDOFF','bit_exact_historical':False,
         'qpos_max_abs_rad':maxrad,'qpos_max_abs_deg':maxdeg,'root_translation_max_abs_m':maxroot,
         'existing_numeric_gate':'configs/gates.v2.yaml:numeric.hinge_roundtrip_max_deg/root_translation_roundtrip_max_m',
         'existing_hinge_roundtrip_max_deg':numeric['hinge_roundtrip_max_deg'],'existing_root_roundtrip_max_m':numeric['root_translation_roundtrip_max_m'],
         'gate_sha256':sha(gates),'diagnostic_sha256':sha(diagnostic),'ground_diagnostic_sha256':sha(ground_diag) if ground_diag else None,
         'diagnosis':'Unchanged ProxQP rate and ground projections on identical full inputs also vary in last bits during isolated serial replays. Original input/solver/thresholds remain unchanged.',
         'different_qa_scalars':list(verified.values()),'training_artifacts_and_decisions_exact':True,
         'new_solver_or_hard_qa_tolerance':False}
