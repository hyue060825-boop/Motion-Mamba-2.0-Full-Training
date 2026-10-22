from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,time,hashlib,zipfile
R=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'));D=R/'delivery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((D/'COMPLETE.json').read_text())
for n,h in c['files'].items():assert sha(D/n)==h,n
compile_receipt=json.loads((D/'MANUSCRIPT_COMPILE.json').read_text());assert compile_receipt['passed'] and compile_receipt['source_sha256']==sha(D/'main.tex')
c['files']={p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name!='COMPLETE.json'};c['local_manuscript_and_visual_QA']=True;c['time']=time.time()
(D/'COMPLETE.json').write_text(json.dumps(c,indent=2)+'\n')
with zipfile.ZipFile(R/'Unified_Easy_V3_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in D.iterdir():
  if p.is_file():z.write(p,'Unified_Easy_V3/'+p.name)
a=dict(passed=True,zip_sha256=sha(R/'Unified_Easy_V3_Results.zip'),delivery_complete_sha256=sha(D/'COMPLETE.json'),conditions=36,formal_rollouts=13824,baseline_checkpoints=8,ours='Native100k',training_or_simulation_changes=False,history_causal_ablation_complete=False,main_tex_compiled=True,time=time.time())
(R/'FINAL_DELIVERY.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
