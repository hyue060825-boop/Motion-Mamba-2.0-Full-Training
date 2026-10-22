"""Replay the unchanged ground QP on complete saved required-clearance series."""
import json
import numpy as np
from g1_pipeline.common import ROOT,write_json,sha,now
from g1_pipeline.ground_clearance import smooth_lift


def run():
 panel=json.loads((ROOT/'audit/parallel_throughput_panel.json').read_text())
 refs={t['row']['sample_id']:t['reference'] for t in panel['tasks']}
 current={r['sample_id']:r for r in map(json.loads,(ROOT/'work/benchmark_throughput_w16_v1/results.jsonl').read_text().splitlines())}
 reports=[]
 for sid in ['hiphi:Cause_motion-hit_0043','hiphi:Cause_motion-hit_0044']:
  pa=ROOT/refs[sid]['metrics_path'];pb=ROOT/current[sid]['metrics_path']
  a=json.loads(pa.read_text())['ground_correction'];b=json.loads(pb.read_text())['ground_correction']
  assert a['original_penetration_m']==b['original_penetration_m']
  required=np.asarray(a['original_penetration_m'])-a['permitted_penetration_m']
  outputs=[];trials=[]
  for offset in [0,8,24,40]:
   memory=np.empty(required.nbytes+64,dtype=np.uint8)
   values=np.ndarray(required.shape,dtype=np.float64,buffer=memory,offset=offset);values[:]=required
   lift=smooth_lift(values,dt=.04,velocity_weight=a['velocity_weight'],acceleration_weight=a['acceleration_weight'])
   outputs.append(lift)
   trials.append({'address_mod64':int(values.ctypes.data%64),'same_input':bool(np.array_equal(values,required)),
                  'max_abs_to_historic_lift_m':float(np.max(abs(lift-a['lift_m']))),
                  'max_abs_to_worker_lift_m':float(np.max(abs(lift-b['lift_m'])))})
  reports.append({'sample_id':sid,'frames':len(required),'trials':trials,
                  'serial_repeat_max_abs_m':max(float(np.max(abs(x-y))) for x in outputs for y in outputs),
                  'historic_worker_lift_max_abs_m':float(np.max(abs(np.asarray(a['lift_m'])-b['lift_m']))),
                  'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [pa,pb]}})
 result={'time':now(),'status':'LOCALIZED_TO_EXISTING_GROUND_QP_ROUNDOFF','results':reports,
         'scope':'Complete saved clearance inputs; unchanged original ProxQP settings; no GMR/solver/QA/input edits',
         'ground_code_sha256':sha(ROOT/'g1_pipeline/ground_clearance.py'),'original_eps_abs':1e-9,'original_eps_rel':1e-9}
 write_json(ROOT/'audit/parallel_ground_roundoff_diagnostic.json',result)
 print({r['sample_id']:{k:r[k] for k in ['frames','serial_repeat_max_abs_m','historic_worker_lift_max_abs_m']} for r in reports})
if __name__=='__main__':run()
