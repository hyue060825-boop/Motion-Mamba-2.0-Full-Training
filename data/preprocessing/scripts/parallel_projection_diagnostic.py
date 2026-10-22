"""Read-only cause localization of historical QP roundoff on a complete clip."""
import json
import numpy as np
from g1_pipeline.common import ROOT,write_json,sha,now
from g1_pipeline.bounded_hinge_trajectory import project,sonic_training_reference_limits
from g1_pipeline.robot import compile_target


def run():
 panel=json.loads((ROOT/'audit/parallel_equivalence_panel.json').read_text())
 task=next(t for t in panel['tasks'] if t['row']['sample_id']=='hiphi:Cause_motion-hit_0042')
 ref=task['reference'];base=(ROOT/ref['output_paths']['qpos_path']).parent
 original=np.load(base/'unprocessed_qpos.npy');historic=np.load(base/'qpos.npy')
 entries=[json.loads(l) for l in (ROOT/'work/benchmark_equivalence_w1_v1/results.jsonl').read_text().splitlines() if l]
 candidate=np.load(ROOT/next(r for r in entries if r['sample_id']==ref['sample_id'])['output_paths']['qpos_path'])
 assert np.array_equal(original,np.load((ROOT/next(r for r in entries if r['sample_id']==ref['sample_id'])['output_paths']['qpos_path']).parent/'unprocessed_qpos.npy'))
 m,asset=compile_target();results=[];saved=[]
 for offset in range(0,64,8):
  memory=np.empty(original.nbytes+64,dtype=np.uint8)
  values=np.ndarray(original.shape,dtype=original.dtype,buffer=memory,offset=offset);values[:]=original
  projected,info=project(values[:,7:],m.jnt_range[1:,0],m.jnt_range[1:,1],sonic_training_reference_limits(m),dt=.04,correction_curvature=1.)
  result={'input_address_mod64':int(values.ctypes.data%64),'input_strides':list(values.strides),
          'input_values_exact':bool(np.array_equal(values,original)),
          'historic_max_abs_rad':float(np.max(abs(projected-historic[:,7:]))),
          'worker_max_abs_rad':float(np.max(abs(projected-candidate[:,7:]))),
          'matches_historic_exactly':bool(np.array_equal(projected,historic[:,7:])),
          'matches_worker_exactly':bool(np.array_equal(projected,candidate[:,7:])),
          'rate_report_matches_historic':info==json.loads((base/'solver.json').read_text())['rate_projection']}
  saved.append(projected);results.append(result)
 report={'time':now(),'scope':'Isolated replay of unchanged rate projection on complete saved pre-postprocessing trajectory. No solver/input/threshold edits.',
         'sample_id':ref['sample_id'],'frames':len(original),'trials':results,
         'max_pairwise_output_difference_rad':max(float(np.max(abs(a-b))) for a in saved for b in saved),
         'original_qp_eps_abs':1e-9,'original_qp_eps_rel':1e-9,'original_qp_duality_gap_abs':1e-10,
         'bindings':{str(base.relative_to(ROOT)/n):sha(base/n) for n in ['unprocessed_qpos.npy','qpos.npy','solver.json']},
         'projection_code_sha256':sha(ROOT/'g1_pipeline/bounded_hinge_trajectory.py')}
 write_json(ROOT/'audit/parallel_projection_roundoff_diagnostic.json',report)
 print({'max_pairwise':report['max_pairwise_output_difference_rad'],'trials':results})
if __name__=='__main__':run()
