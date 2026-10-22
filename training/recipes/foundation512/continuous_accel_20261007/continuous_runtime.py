"""Execution-only continuity: retain the complete recipe and save machinery."""
import ast
import copy
import hashlib
import inspect
import os
import time
from pathlib import Path

VERSION = 'continuous_checkpoints_v1'


def make_train(engine):
    source = inspect.getsource(engine.train)
    tree = ast.parse(source)
    expected = ast.parse("step in cfg['savepoints'] or step==until", mode='eval').body
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
               and ast.dump(n.test) == ast.dump(expected)]
    assert len(matches) == 1, 'Unexpected engine save condition'
    matches[0].test = ast.parse("step in cfg['savepoints'] or step==until or step%1000==0", mode='eval').body
    ast.fix_missing_locations(tree)
    # Reversing the sole transformation must reproduce the original AST.
    reverted = copy.deepcopy(tree)
    changed = [n for n in ast.walk(reverted) if isinstance(n, ast.If)
               and ast.dump(n.test) == ast.dump(matches[0].test)]
    assert len(changed) == 1
    changed[0].test = expected
    assert ast.dump(reverted) == ast.dump(ast.parse(source))
    # Compile again in the live module namespace so installed update/save are used.
    exec(compile(tree, str(Path(__file__).with_name('generated_train.py')), 'exec'), engine.__dict__)
    return dict(original_train_sha256=hashlib.sha256(source.encode()).hexdigest(),
                transformed_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
                only_change='add step%1000==0 to existing save condition',
                generated_source=ast.unparse(tree))


def install(engine, root, evidence):
    from common import read, write, OLD
    from motion_training import runtime, distributed as parallel
    from motion_training.io import identity
    root = Path(root)
    audit = make_train(engine)
    assert audit['original_train_sha256'] == evidence['original_train_sha256']
    assert audit['transformed_ast_sha256'] == evidence['transformed_ast_sha256']
    prior_save = engine.save
    if evidence['optimizer_copy_enabled']:
        from optimizer_copy import install_optimizer_copy
        original_optimizer = runtime.optimizer

        def optimizer(*args, **kwargs):
            opt = original_optimizer(*args, **kwargs)
            install_optimizer_copy(opt)
            return opt

        runtime.optimizer = optimizer

    def save(path, m, ck, opt, cfg, step, manifest, pool):
        ck['continuous_execution'] = dict(evidence, source=str(Path(__file__)),
                                           recipe_and_sampler_unchanged=True)
        started = time.perf_counter()
        prior_save(path, m, ck, opt, cfg, step, manifest, pool)
        # Publish only after the leader has atomically finished the checkpoint.
        parallel.barrier()
        if parallel.leader():
            global_step = ck['_framework_global_start'] + step
            state = dict(status='SAVED_CONTINUING', step=step, checkpoint=str(path),
                         recipe_hash=identity(cfg), model_qualified=False)
            write(root/'train/STATE.json', state)
            link = OLD/'train'/f'checkpoint{global_step}.pt'
            if not link.exists():
                link.symlink_to(Path(path).resolve())
            write(root/'LATEST.json', dict(global_step=global_step, stage_step=step,
                  checkpoint=str(path), previous_exposures=2400000,
                  additional_exposures=step*512, total_exposures=2400000+step*512,
                  new_recipe=True, original_75k_preserved=True))
            old_status = read(root/'STATUS.json')
            old_status.update(stage='TRAINING_BATCH512_CONTINUOUS', time=time.time(),
                              global_step=global_step, target_global_step=100000,
                              continuous_worker=True, checkpoint=str(path))
            write(root/'STATUS.json', old_status)
            with (Path(__file__).parent/'SAVE_EVENTS.jsonl').open('a') as f:
                import json
                f.write(json.dumps(dict(stage_step=step, global_step=global_step,
                     checkpoint=str(path), save_seconds=time.perf_counter()-started,
                     worker_pid=os.getpid(), time=time.time()))+'\n')

    engine.save = save
    return audit
