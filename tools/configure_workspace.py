"""Create an explicit ignored artifact map; no downloading, moving or overwriting assets."""
from pathlib import Path
import argparse,json,os,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output',default='artifacts.local.json');p.add_argument('--bind',action='append',default=[],metavar='KEY=PATH');a=p.parse_args()
out=Path(a.output);assert not out.exists(),'Edit an existing local map deliberately; never overwrite it'
ws=Path(a.workspace).expanduser().resolve();ws.mkdir(parents=True,exist_ok=True)
for name in ['data','checkpoints','runs','artifacts','deps','private_archive']:(ws/name).mkdir(exist_ok=True)
values={}
for item in a.bind:
 key,path=item.split('=',1);assert key in json.loads((R/'configs/artifacts.example.json').read_text()),key
 path=Path(os.path.expandvars(path)).expanduser().resolve();assert path.exists(),path;values[key]=dict(path=str(path),required_external=True)
out.write_text(json.dumps(values,indent=2)+'\n');print('WORKSPACE',ws,'LOCAL_MAP',out.resolve())
