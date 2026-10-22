"""Explicit reproducible foundation entry; no private server paths or automatic GPU queue."""
from pathlib import Path
import argparse,sys,json,os
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'training'),str(R/'data')]
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['foundation32','foundation512'],required=True);p.add_argument('--config');p.add_argument('--output',required=True);p.add_argument('--until',type=int,required=True);p.add_argument('--resume');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
from motion_training.io import read,write
cfgpath=Path(a.config or R/'configs/training'/f'{a.phase}.json');cfg=read(cfgpath)
if a.dry_run:
 print(json.dumps(dict(phase=a.phase,config=cfg,output=a.output,until=a.until,resume=a.resume,training_started=False),indent=2));raise SystemExit
import motion_training
motion_training.__path__.insert(0,str(R/'training/revisions'/('foundation' if a.phase=='foundation32' else 'foundation512')))
from motion_training import engine,data,batching,streaming,distributed as parallel
parallel.setup();batching.install();engine.update=batching.update
if a.phase=='foundation32':
 data.prepare=streaming.prepare;engine.read=lambda path:streaming.load(path) if Path(path).name=='MANIFEST.json' else read(path)
else:
 sys.path.insert(0,str(R/'training/recipes/foundation512'));import lazy_plan
 data.prepare=lazy_plan.prepare;engine.read=lambda path:lazy_plan.load(path) if Path(path).name=='MANIFEST.json' else read(path)
engine.train(str(cfgpath),a.output,until=a.until,resume=a.resume,category='B')
