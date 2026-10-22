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

# The opt-in execution backend is outside the model/recipe source snapshot.
# Its exact source digest and admission reports are persisted in checkpoints.
execution_root=ROOT/'codec_graph_accel_20261007'
if (execution_root/'ENABLED.json').exists():
    import hashlib
    execution=read(execution_root/'ENABLED.json')
    assert hashlib.sha256((execution_root/'pipeline_accel.py').read_bytes()).hexdigest()==execution['source_sha256']
    assert read(execution['gate'])['passed']
    assert hashlib.sha256((execution_root/'codec_graph.py').read_bytes()).hexdigest()==execution['codec_source_sha256']
    sys.path.insert(0,str(execution_root))
    import pipeline_accel
    assert pipeline_accel.VERSION==execution['version']
    assert Path(pipeline_accel.__file__).resolve()==(execution_root/'pipeline_accel.py').resolve()
    engine.update=pipeline_accel.install(execution['mode'],execution['workers'])
    original_save=engine.save
    def execution_save(path,m,ck,opt,cfg,step,manifest,pool):
        ck['execution_acceleration']=dict(execution,backend_source=str(execution_root/'pipeline_accel.py'))
        return original_save(path,m,ck,opt,cfg,step,manifest,pool)
    engine.save=execution_save

engine.train(str(ROOT/'config.json'),str(ROOT/'train'),until=a.until,resume=a.resume,category='B')
