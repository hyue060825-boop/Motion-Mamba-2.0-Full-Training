"""Inspect source/artifact prerequisites without starting training or selecting a GPU."""
from pathlib import Path
import argparse,json,ast,hashlib,os,sys,importlib.metadata
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src'),str(R/'training'),str(R/'data'),str(R/'evaluation')]
from motion_mamba2.paths import bindings,artifact_path
p=argparse.ArgumentParser();p.add_argument('--strict-assets',action='store_true');p.add_argument('--config',help='Check only artifacts referenced by this recipe when strict');a=p.parse_args()
errors=[];count=0
for f in R.rglob('*.py'):
 if any(v in f.parts for v in ['.git','.venv','__pycache__']):continue
 try:ast.parse(f.read_text());count+=1
 except SyntaxError as e:errors.append(dict(file=str(f.relative_to(R)),error=str(e)))
keys=list(bindings())
if a.config:
 import re
 keys=sorted(set(re.findall(r'artifact://([a-zA-Z0-9_.-]+)',Path(a.config).read_text())))
assets=[dict(key=k,path=artifact_path(k),exists=Path(artifact_path(k)).exists()) for k in keys]
packages={}
for n in ['torch','numpy','triton','mamba-ssm','transformers','safetensors','hydra-core']:
 try:packages[n]=importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:packages[n]=None
out=dict(source_syntax_passed=not errors,source_files=count,syntax_errors=errors,packages=packages,bound_assets=assets,training_started=False,GPU_claimed=False)
print(json.dumps(out,indent=2));raise SystemExit(1 if errors or (a.strict_assets and any(not x['exists'] for x in assets)) else 0)
