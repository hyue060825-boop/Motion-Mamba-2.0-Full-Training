"""Explicit metadata-only migration of a trusted full checkpoint; original is never overwritten."""
from pathlib import Path
import argparse,copy,json,sys,hashlib,os
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'training')]
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('--recipe',required=True);a=p.parse_args()
import torch
from motion_mamba2.legacy_compat import checkpoint_metadata,target_namespace
from motion_mamba2.paths import resolve_tree
from motion_training.io import read,identity,write,sha
torch.set_num_threads(2);source=Path(a.source);dest=Path(a.destination);assert not dest.exists() and source.resolve()!=dest.resolve()
x=torch.load(source,map_location='cpu',weights_only=False,mmap=True);y=checkpoint_metadata(x);framework=copy.deepcopy(x.get('framework',{}));assert framework.get('effective_recipe'),'Full framework checkpoint required'
old=framework['effective_recipe'];new=read(a.recipe);migrated=resolve_tree(target_namespace(old))
assert migrated==new,'Recipe differs beyond explicit namespace/path metadata migration; do not bypass resume admission'
state_keys=[k for k in ['root_model','body_model','denoiser_aux','model','optimizer','rng_by_rank','rng'] if k in x]
before={k:identity(x[k]) for k in state_keys};framework['effective_recipe']=new;framework['recipe_hash']=identity(new)
for k in ['manifest_path','pool_path']: 
 if framework.get(k):framework[k]=resolve_tree(framework[k])
y['framework']=framework
y['release_metadata_migration']=dict(source_sha256=sha(source),original_recipe_hash=x['framework']['recipe_hash'],new_recipe_hash=framework['recipe_hash'],tensor_state_hashes=before,source_admission_transferred=False,scope='Paths/namespace only; numerical recipe equality required; re-admit GPU acceleration')
assert all(identity(y[k])==h for k,h in before.items());dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');torch.save(y,tmp);tmp.replace(dest)
z=torch.load(dest,map_location='cpu',mmap=True,weights_only=False);assert all(identity(z[k])==h for k,h in before.items())
write(dest.with_suffix('.migration.json'),dict(passed=True,source_sha256=sha(source),destination_sha256=sha(dest),preserved_state_hashes=before,original_unchanged=True))
print('METADATA_MIGRATION_VERIFIED',dest)
