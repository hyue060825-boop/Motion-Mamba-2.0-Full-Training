from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,os,hashlib
ROOT=Path(__file__).resolve().parent
PY=artifact_path('python_9f8003379b')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p)
 if p.name=='INFERENCE_TRACE.json':return # Same exclusion as the frozen parent, including warmed timing.
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
