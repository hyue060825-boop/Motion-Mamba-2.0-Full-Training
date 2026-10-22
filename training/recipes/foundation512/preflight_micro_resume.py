"""CPU exercise of the exact resume validation without altering a checkpoint."""
import ast,textwrap,torch
from common import *
from motion_training import engine
from motion_training.io import identity,sha
parent=ROOT/'train/checkpoint100.pt'
ck=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)
old=engine.config(ROOT/'config.json');assert old['per_rank_microbatch']==16
assert identity(old)==ck['framework']['recipe_hash']
new=dict(old,per_rank_microbatch=64)
gate=dict(passed=True,kind='micro16_to64_same_global512',resume_checkpoint=str(parent),resume_sha256=sha(parent),old_recipe_hash=identity(old),new_recipe_hash=identity(new))
template=ast.parse((ROOT/'install_micro_gate.py').read_text())
body=next(ast.literal_eval(n.value) for n in template.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='new' for t in n.targets))
namespace=dict(os=os,read=lambda _:gate,sha=sha,recipe_hash=engine.recipe_hash)
exec('def validate(ck,cfg,cp):\n'+textwrap.indent(textwrap.dedent(body),'    '),namespace)
os.environ['MOTION_MICROBATCH_GATE']='cpu_probe_gate'
namespace['validate'](ck,new,parent)
assert ck['_resource_transition']['optimizer_and_scheduler_inherited']
bad=dict(new,lr=new['lr']*2)
try:namespace['validate'](ck,bad,parent)
except AssertionError:pass
else:raise AssertionError('Unrelated learning-rate change was not rejected')
write(ROOT/'micro64_switch/RESUME_CPU_PREFLIGHT.json',dict(passed=True,exact_installer_validation_exercised=True,
    canonical_recipe_hash_matches_saved_checkpoint=True,only_microbatch_change_allowed=True,unrelated_lr_change_rejected=True,
    checkpoint_not_modified=True,gpu_not_used=True))
print('MICRO64_RESUME_PREFLIGHT_PASS')
