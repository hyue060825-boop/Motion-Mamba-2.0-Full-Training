"""Bind compatible existing metadata to a new frozen panel without inventing provenance."""
import argparse
from pathlib import Path
from .common import rows,read,jsonl,write,sha
from .v2_score import verify_release
from .schema import load_sample

def migrate(manifest,release,metric,output):
    folder=Path(release);r=verify_release(folder);name={'retrieval':'retrieval_panel.json','fid':'fid52_panel.json','both':'combined_panel.json'}[metric];p=read(folder/name);metas=list(rows(manifest));by={m['sample_id']:m for m in metas};out=[]
    if len(by)!=len(metas) or set(by)!={m['sample_id'] for m in p['records']}:raise ValueError('Old manifest does not exactly cover new panel; generate missing samples, do not silently filter')
    sk=read(folder/'SKELETON.json')
    for expected in p['records']:
        old=by[expected['sample_id']];a,(s,e)=load_sample(old,Path(manifest).parent,sk['sha256'])
        if old['dataset']!=r['domain'] or old['source_family_id']!=expected['source_family_id'] or e-s!=expected['frames']:raise ValueError('Old data incompatible with new frozen panel')
        array=Path(old['array_path']);array=array if array.is_absolute() else Path(manifest).parent/array
        out.append(dict(old,array_path=str(array.resolve()),track=r['track'],panel_id=p['panel_sha256']))
    jsonl(output,out);write(Path(output).with_suffix('.migration.json'),dict(source_sha256=sha(manifest),release_sha256=r['release_sha256'],changes=['track','panel_id','absolute array path'],generation_identity_changed=False))
    return dict(samples=len(out),output=str(output))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--release',required=True);p.add_argument('--metric',choices=['retrieval','fid','both'],required=True);p.add_argument('--output',required=True);a=p.parse_args();print(migrate(a.samples,a.release,a.metric,a.output))
