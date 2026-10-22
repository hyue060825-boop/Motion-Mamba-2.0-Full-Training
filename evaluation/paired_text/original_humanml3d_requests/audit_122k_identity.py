"""Read-only audit separating the two122k checkpoints and their result lineage."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,json,hashlib,time,csv,math
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
R=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'));H=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
O=R/'diagnosis/native122_identity_20261021';O.mkdir(parents=True,exist_ok=True)
def rd(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
reg=rd(R/'CHECKPOINTS.json');identities={}
for key in ['native122','native122long']:
 v=reg[key];p=Path(v['checkpoint']);actual=sha(p);assert actual==v['sha256']
 identities[key]=dict(checkpoint=str(p),resolved_path=str(p.resolve()),sha256=actual,size=p.stat().st_size)
assert identities['native122']['sha256']!=identities['native122long']['sha256']
assert identities['native122']['resolved_path']!=identities['native122long']['resolved_path']
loads=[]
for t in ['short52','nominal9','nominal20','nominal40']:
 d=R/'models/native122'/t;l=rd(d/'LOAD.json');c=rd(d/'GENERATION_COMPLETE.json');scored=rd(R/'scoring/models/native122'/t/'COMPLETE.json')
 assert l['checkpoint']==identities['native122']['checkpoint'] and l['global_step']==122000 and l['all_loaded_denoiser_weights_verified']
 assert c['checkpoint']['sha256']==identities['native122']['sha256'] and c['checkpoint']['checkpoint']==l['checkpoint']
 assert c['rollouts']==384 and scored['passed']
 loads.append(dict(task=t,checkpoint=l['checkpoint'],all_loaded_weights_equal_checkpoint=True,global_step=l['global_step'],recorded_checkpoint_sha256=c['checkpoint']['sha256'],load_record_sha256=sha(d/'LOAD.json'),generation_record_sha256=sha(d/'GENERATION_COMPLETE.json'),score_complete_sha256=sha(R/'scoring/models/native122'/t/'COMPLETE.json')))
old=[]
for key in ['native122','native122long']:
 count=0
 for p in sorted((H/'models'/key).rglob('LOAD.json')):
  a=rd(p)
  if a.get('checkpoint'):
   assert a['checkpoint']==identities[key]['checkpoint'],('old_label_binding_mismatch',key,str(p),a['checkpoint'])
   count+=1
   if count<=2:old.append(dict(label=key,path=str(p),checkpoint=a['checkpoint'],weights_verified=a.get('all_loaded_denoiser_weights_verified')))
d=R/'scoring/models/native122/nominal40';rows=list(csv.DictReader((d/'PER_TRIAL.csv').open()));examples=[]
cases=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test']
for row in sorted(rows,key=lambda v:float(v['root_CE_cm']),reverse=True)[:3]:
 i=int(row['case']);si=int(row['seed_index']);p=R/'models/native122/nominal40/trials'/f'{i:03d}_s{si}'/'geometry.npz'
 with np.load(p) as z:root=z['root'][-1,[0,2]].astype(float)
 target=np.asarray(cases[i]['root_goal_xz_m'],float);delta=root-target;ce=math.hypot(float(delta[0]),float(delta[1]))*100
 assert abs(ce-float(row['root_CE_cm']))<1e-9
 examples.append(dict(case=i,seed_index=si,final_root_xz_m=root.tolist(),goal_xz_m=target.tolist(),independent_CE_cm=ce,stored_CE_cm=float(row['root_CE_cm'])))
out=dict(passed=True,two_actual_file_hashes_recomputed=True,two_resolved_paths_distinct=True,evaluated_version='native122_regular_continue5k',evaluated_long_distill=False,identities=identities,all_four_loaded_weight_bindings=loads,old_HumanML3D_labels_checked=old,three_large_error_examples_independently_recomputed=examples,scope='Checkpoint identity and stored-result lineage verified; does not alone exclude every possible evaluation-interface issue',time=time.time())
(O/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
