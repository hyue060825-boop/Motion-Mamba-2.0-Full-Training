from motion_mamba2.paths import artifact_path
from pathlib import Path
p=Path(artifact_path('resource_switch_py_1511f7818c'));s=p.read_text();backup=p.with_suffix('.py.before_batch32_switch');assert not backup.exists();backup.write_text(s)
pos=s.index('    gate=read(path)',s.index('def validate_transition'))+len('    gate=read(path)')
branch='''
    if cfg['batch_size']==32:
        assert gate['kind']=='batch32_resource_transition' and gate['passed']
        assert gate['from_ranks']==old_world==2 and gate['to_ranks']==new_world==4
        assert gate['effective_batch']==32 and cfg['gt_slots']==32
        assert gate['resume_checkpoint']==str(Path(checkpoint).resolve())
        assert gate['resume_sha256']==sha(checkpoint)
        assert gate['recipe_identity']==identity(cfg)
        assert gate['engine_sha256']==sha(Path(__file__).with_name('engine.py'))
        assert gate['batching_sha256']==sha(Path(__file__).with_name('batching.py'))
        return dict(kind='resource_only_resume',from_ranks=2,to_ranks=4,gate=str(Path(path).resolve()),
            recipe_unchanged=True,effective_batch=32,per_rank_batch=8,
            optimizer_master_scheduler_inherited=True,
            rng_policy='saved rank modulo old world; each physical sample/noise/dropout seed unchanged',
            numerical_policy='validated update tolerance, not bit-identical across world sizes')
'''
p.write_text(s[:pos]+branch+s[pos:])
