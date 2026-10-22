"""Explicit portable artifact bindings. Missing assets fail rather than using a private server."""
from pathlib import Path
import os,json,re
from functools import lru_cache

def workspace():return Path(os.environ.get('MM2_WORKSPACE','./workspace')).expanduser().resolve()
@lru_cache(maxsize=8)
def _load_bindings(path):return json.loads(Path(path).expanduser().read_text()) if path else {}
def bindings():
 default=Path(__file__).resolve().parents[2]/'configs/artifacts.example.json'
 base=dict(_load_bindings(str(default))) if default.exists() else {}
 base.update(_load_bindings(os.environ.get('MM2_ARTIFACT_MAP','')))
 return base
@lru_cache(maxsize=8)
def _prefixes(path):
 return sorted(_load_bindings(path).items(),key=lambda z:len(z[1].get('legacy_path','')) if isinstance(z[1],dict) else 0,reverse=True)
def artifact_path(key):
 value=bindings().get(key,{})
 value=value.get('path') if isinstance(value,dict) else value
 if not value:return str(workspace()/'artifacts'/key)
 return os.path.expandvars(os.path.expanduser(value)).replace('${MM2_WORKSPACE}',str(workspace())).replace('${MM2_REPOSITORY}',str(Path(__file__).resolve().parents[2]))
def resolve_tree(value):
 if isinstance(value,dict):return {k:resolve_tree(v) for k,v in value.items()}
 if isinstance(value,list):return [resolve_tree(v) for v in value]
 if isinstance(value,tuple):return tuple(resolve_tree(v) for v in value)
 if not isinstance(value,str):return value
 if value.startswith('artifact://'):return artifact_path(value[11:])
 if not value.startswith('/'):return value
 for key,entry in _prefixes(os.environ.get('MM2_ARTIFACT_MAP','')):
  if isinstance(entry,dict) and entry.get('legacy_path'):
   old=entry['legacy_path']
   if value==old or value.startswith(old.rstrip('/')+'/'):return artifact_path(key)+value[len(old):]
 return value
