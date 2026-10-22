import argparse
from common import *
from motion_training import engine,data,batching,distributed as parallel
from motion_training.io import read as original_read
import lazy_plan
parallel.setup();batching.install();engine.update=batching.update;data.prepare=lazy_plan.prepare
engine.read=lambda path:lazy_plan.load(path) if Path(path).name=='MANIFEST.json' else original_read(path)
p=argparse.ArgumentParser();p.add_argument('--until',type=int,required=True);p.add_argument('--resume');a=p.parse_args()
# Same effective recipe after a four->eight switch. The global batch, schedule,
# and sampler cursor remain unchanged; only rank partitioning is different.
cfg=read(ROOT/'config.json')
if a.resume and parallel.world()!=__import__('torch').load(a.resume,map_location='cpu',mmap=True,weights_only=False)['framework']['world_size']:
    gate=read(ROOT/'TOPOLOGY_GATE.json');assert gate['passed']
    import torch
    from motion_training.io import sha,identity
    saved=torch.load(a.resume,map_location='cpu',mmap=True,weights_only=False)
    assert gate['resume_checkpoint']==a.resume and gate['resume_sha256']==sha(a.resume)
    def admit(path,recipe,checkpoint,old_world,new_world):
        assert path==str(ROOT/'TOPOLOGY_GATE.json') and old_world==4 and new_world==8
        assert recipe['batch_size']==512 and gate['recipe_hash']==identity(cfg)
        assert identity(recipe)==identity(engine.config(ROOT/'config.json'))
        return dict(kind='same_global_batch512_topology_switch',from_ranks=4,to_ranks=8,
                    gate=path,optimizer_master_scheduler_inherited=True,global_batch=512,
                    rng_policy='saved rank modulo old world; frozen per-record noise unchanged')
    from motion_training import resource_switch
    resource_switch.validate_transition=admit
engine.train(str(ROOT/'config.json'),str(ROOT/'train'),until=a.until,resume=a.resume,category='B')
