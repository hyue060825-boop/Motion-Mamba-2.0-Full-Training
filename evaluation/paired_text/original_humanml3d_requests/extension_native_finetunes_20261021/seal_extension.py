from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,time,hashlib,zipfile
R=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'));E=R/'extension_native_finetunes_20261021';D=E/'delivery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((D/'COMPLETE.json').read_text());assert c['passed']
for name,h in c['files'].items():
 if name!='RESULTS.md':assert sha(D/name)==h,name
a=json.loads((D/'MANUSCRIPT_COMPILE.json').read_text());assert a['passed'] and a['source_sha256']==sha(D/'main.tex')
c.update(local_manuscript_compile_pending=False,local_verification_complete=True,time=time.time());c['files']={p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
(D/'COMPLETE.json').write_text(json.dumps(c,indent=2)+'\n')
with zipfile.ZipFile(E/'Unified_Easy_V3_All12_With_Native_Finetunes.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in D.iterdir():
  if p.is_file():z.write(p,'Unified_Easy_V3_All12/'+p.name)
f=json.loads((E/'FINAL_DELIVERY.json').read_text());f.update(zip_sha256=sha(E/'Unified_Easy_V3_All12_With_Native_Finetunes.zip'),complete_sha256=sha(D/'COMPLETE.json'),local_manuscript_compilation_pending=False,local_verification_complete=True,time=time.time());(E/'FINAL_DELIVERY.json').write_text(json.dumps(f,indent=2)+'\n')
(E/'STATUS.json').write_text(json.dumps(dict(stage='ADDITIONAL_NATIVE_CHECKPOINTS_EVALUATED_AND_DELIVERED',new_conditions=12,combined_conditions=48,new_formal_rollouts=4608,combined_formal_rollouts=18432,delivery=str(D),history_causal_ablation_complete=False,time=time.time()),indent=2)+'\n')
p=R/'EXTENSION_ACTIVE.json';x=json.loads(p.read_text());x.update(active=False,stage='COMPLETED_AND_DELIVERED',final_delivery=str(E/'FINAL_DELIVERY.json'),time=time.time());p.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(f))
