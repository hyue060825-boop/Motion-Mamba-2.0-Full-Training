from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,shutil,ast,hashlib,py_compile
R=Path(artifact_path('g1_native100k_original117k_protocol_20261014_dbcf4a1353'))
E=Path(artifact_path('engine_6fc02cd60c'))
def replace(s,a,b):
 assert a in s,a[:120]
 return s.replace(a,b)
s=(R/'reference/train_future_117k.py').read_text()
s=replace(s,'import lazy_plan, speed_backend as speed','import lazy_plan, speed_backend as speed\nimport extreme_runtime')
s=replace(s,'with prediction_loss_context(self.raw,pred):value,info,_=fast.components_group(self.raw,f,pred)',
 '''with extreme_runtime.positions_only(self.raw):
            with prediction_loss_context(self.raw,pred):value,info,_=fast.components_group(self.raw,f,pred)''')
s=s.replace('parallel.world()==8','parallel.world()==4').replace("saved['world_size']==8","saved['world_size']==4")
s=s.replace("admission['world_size']==8","admission['world_size']==4")
s=s.replace('112000','100000')
s=replace(s,"opt=runtime.optimizer(m,ck,cfg,resume=a.start>0)","opt=runtime.optimizer(m,ck,cfg if a.start>0 else dict(cfg,scheduler=ck['framework']['scheduler']['spec']),resume=a.start>0)")
s=replace(s,"if a.gate_only:\n    from motion_training.io import Budget","extreme_runtime.install(c)\nif a.gate_only:\n    from motion_training.io import Budget")
s=replace(s,"dict(physical_world_size=8,global_batch=512,GT_per_rank=56,teacher_per_rank=8,", "dict(physical_world_size=4,logical_lanes=8,global_batch=512,GT_per_rank=112,teacher_per_rank=16,")
s=s.replace('stage_start=a.start,world_size=8','stage_start=a.start,world_size=4')
s=replace(s,'rs=choose(pool,step,parallel.rank(),cfg)','lanes=[parallel.rank()*2,parallel.rank()*2+1]\n    rs=[r for lane in lanes for r in choose(pool,step,lane,cfg)]')
start=s.index('    records=plan.records_at(step-1,448);subset=')
end=s.index('    values=[];teacher_receipts=[]',start)
s=s[:start]+'''    extreme_runtime.enable(True)
    records=plan.records_at(step-1,448)
    local=[r for lane in lanes for r in records[lane*56:(lane+1)*56]]
    pending=fast.submit_reads(c,local,cfg,2);gt_values=[]
    for micro,lane in enumerate(lanes):
        subset=local[micro*56:(micro+1)*56]
        prepared=fast.materialize(pending[micro*56:(micro+1)*56])
        f,gt_receipts=fast.build_group(m,c,prepared,cfg,audit=audit,serial_codec=True)
        torch.manual_seed(subset[0]['seed']);torch.cuda.manual_seed_all(subset[0]['seed'])
        with module.no_sync():
            with torch.autocast('cuda',dtype=torch.bfloat16):v,info=module(f)
            assert torch.isfinite(v);(v*(4*56/512)).backward()
        gt_values.append(v.detach());del v,f,prepared
    gt_loss=float(torch.stack(gt_values).mean())
''' +s[end:]
s=s.replace('(v*(8/512)).backward()','(v*(4/512)).backward()')
s=replace(s,"if v5 is not None:v5.end_update()\n    norm=", "if v5 is not None:v5.end_update()\n    extreme_runtime.end()\n    norm=")
s=s.replace('GT_slots=56,teacher_slots=8,future_slots=4,legacy_slots=4','GT_slots=112,teacher_slots=16,future_slots=8,legacy_slots=8')
s=s.replace("world_size=8,sample_slots=512","world_size=4,sample_slots=512")
s=s.replace("world_size=8,global_batch=512","world_size=4,global_batch=512")
s=s.replace("for x in allparts)/8","for x in allparts)/4")
(R/'train_future.py').write_text(s)
def choose_source(source):
 tree=ast.parse(source)
 return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='choose'),include_attributes=False)
assert choose_source(s)==choose_source((R/'reference/train_future_117k.py').read_text())
for f in ['profile_extreme.py','extreme_runtime.py','layer_graph.py','geometry_fast.py','valid_scope.py']:
 shutil.copy2(E/f,R/f)
# New-branch history execution is permitted only after a fresh full512 gate.
for f in ['history_v5.py','extra_graph.py','encode_graph.py','condition_cache.py']:
 shutil.copy2(E/'acceleration_v5'/f,R/'acceleration_v5'/f)
for f in ['train_future.py','profile_extreme.py','controller.py']:
 py_compile.compile(str(R/f),doraise=True)
audit=json.loads((R/'ORIGINAL_PROTOCOL_AUDIT.json').read_text())
for f,h in audit['original_source_hashes'].items():
 assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,(f,'Old recipe source changed')
(R/'TRAINING_FILES_READY.json').write_text(json.dumps(dict(passed=True,original_choose_function_exact=True,
 old_GT_builder_losses_schema_and_future_collector_hashes_unchanged=True,
 physical_ranks=4,logical_lanes=8,GT=448,legacy=32,root_prepare=24,root_arrival=8,
 no_new_FK_or_spatial_or_long_goal_material=True,
 topology_arithmetic='4-rank DDP mean of two logical lanes weighted4/512 equals8-rank mean weighted8/512',
 actual_8GPU_bitwise_equality_not_claimed=True,acceleration_requires_new_pool_full_update_gate=True),indent=2))
print('Frozen worker ready; original8 logical sample lanes and original objective preserved.')
