def train(cfgpath, out, until=None, resume=None, parent=None, pool=None, category=None):
    cfg = config(cfgpath)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    budget = Budget(cfg['budget_root'])
    until = cfg['max_updates'] if until is None else until
    assert 0 <= until <= cfg['max_updates']
    category = category or ('C' if cfg['name'] == 'closed_loop_v1' else 'B')
    if cfg.get('execution_authorization') == 'PLAN_ONLY':
        raise RuntimeError('R3 is planned only; explicit launch authorization and frozen resource limits required')
    if (dest / 'STOP.json').exists():
        raise RuntimeError('Branch stopped; do not resume damaged recipe')
    with parallel.timed(budget, 'train:' + str(dest)):
        manifest = dest / 'data/MANIFEST.json'
        if parallel.leader():
            data.prepare(cfg, manifest.parent)
        parallel.barrier()
        plan = read(manifest)
        cp = Path(resume or parent or cfg['parent'])
        fresh_seed = cfg.get('initialization_seed', cfg['seed']) if not resume and cfg.get('initialization_policy') == 'fresh_denoiser' else None
        m, c, ck = runtime.load(cp, True, fresh_seed=fresh_seed)
        opt = runtime.optimizer(m, ck, cfg, resume=bool(resume))
        start = 0
        if fresh_seed is None:
            parent_rank = 0
            if not resume and cfg.get('restore_parent_rank_rng'):
                assert ck['framework']['world_size'] == parallel.world(), 'Parent rank RNG requires unchanged world size'
                assert len(ck['rng_by_rank']) == parallel.world()
                parent_rank = parallel.rank()
            runtime.restore_rng(ck['rng_by_rank'][parent_rank], fallback_seed=cfg['seed'] if not resume else None)
        if resume:
            saved = ck['framework']
            if saved['recipe_hash'] != recipe_hash(cfg):
                gatepath = os.environ.get('MOTION_MICROBATCH_GATE')
                assert gatepath, 'Recipe changed: explicit admitted microbatch transition required'
                gate = read(gatepath)
                before = saved['effective_recipe']
                assert gate['passed'] and gate['kind'] == 'micro16_to64_same_global512'
                assert str(cp) == gate['resume_checkpoint'] and sha(cp) == gate['resume_sha256']
                assert saved['recipe_hash'] == gate['old_recipe_hash'] and recipe_hash(cfg) == gate['new_recipe_hash']
                diff = {k for k in set(before) | set(cfg) if before.get(k) != cfg.get(k)}
                assert diff == {'per_rank_microbatch'} and before['per_rank_microbatch'] == 16 and (cfg['per_rank_microbatch'] == 64)
                assert before['batch_size'] == cfg['batch_size'] == 512
                ck['_resource_transition'] = dict(kind=gate['kind'], gate=gatepath, global_batch=512, old_micro=16, new_micro=64, optimizer_and_scheduler_inherited=True)
            saved_world = saved.get('world_size', 1)
            if saved_world != parallel.world():
                from .resource_switch import validate_transition
                transition = validate_transition(os.environ.get('MOTION_WORLD_SIZE_GATE'), cfg, cp, saved_world, parallel.world())
                ck['_resource_transition'] = transition
                runtime.restore_rng(ck['rng_by_rank'][parallel.rank() % saved_world])
            else:
                runtime.restore_rng(ck['rng_by_rank'][parallel.rank()])
                if saved.get('resource_transition'):
                    ck['_resource_transition'] = saved['resource_transition']
            assert saved['manifest_hash'] == sha(manifest), 'Sampler plan changed'
            start = saved['stage_step']
            assert len(rows(dest / 'train.jsonl')) == start, 'Receipt/cursor mismatch'
            if not pool:
                pool = saved['pool_path']
            if pool and saved['pool_path'] == str(pool):
                assert saved['pool_hash'] == sha(pool)
        ck['_framework_global_start'] = ck.get('step', 16600) - start
        if category == 'C':
            assert pool, 'Current-history pool required'
            pool = Path(pool)
            pooldata = read(pool)
            assert pooldata['labels']
            if not resume or ck['framework'].get('pool_path') != str(pool):
                assert pooldata['checkpoint_sha256'] == sha(cp), 'Initial/refreshed pool must match the current student checkpoint'
            from .collectors import schedule_pool
            event_plan = schedule_pool(pooldata, until - start, start=start)
        if parallel.leader():
            write(dest / 'EFFECTIVE_CONFIG.json', dict(recipe=cfg, recipe_hash=recipe_hash(cfg), parent=str(cp), parent_sha256=sha(cp), optimizer_groups=[{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups], imports=runtime.import_sources(), stage_start=start, world_size=parallel.world()))
        if start == 0 and (not (dest / 'checkpoint0.pt').exists()):
            save(dest / 'checkpoint0.pt', m, ck, opt, cfg, 0, manifest, pool)
        for step in range(start + 1, until + 1):
            records = plan['plan'][(step - 1) * cfg['gt_slots']:step * cfg['gt_slots']]
            if category == 'C':
                records += event_plan[step - start - 1]
            rec = update(m, c, ck, opt, cfg, records, step, category, budget, gradient_details=step in [1, 32, 64, 128])
            if parallel.leader():
                append(dest / 'train.jsonl', rec)
                print('UPDATE', cfg['name'], step, round(rec['loss'], 5), round(rec['seconds'], 2), flush=True)
            if step in cfg['savepoints'] or step == until or step % 1000 == 0:
                save(dest / f'checkpoint{step}.pt', m, ck, opt, cfg, step, manifest, pool)
            if cfg.get('boundary_request_path') and step % 100 == 0:
                requests = parallel.gather(Path(cfg['boundary_request_path']).exists())
                if any(requests):
                    target = dest / f'checkpoint{step}.pt'
                    if not target.exists():
                        save(target, m, ck, opt, cfg, step, manifest, pool)
                    if parallel.leader():
                        write(dest / 'STATE.json', dict(status='BOUNDARY_REQUEST', step=step, checkpoint=str(target), recipe_hash=recipe_hash(cfg), model_qualified=False))
                    parallel.barrier()
                    return str(target)
        if parallel.leader():
            write(dest / 'STATE.json', dict(status='SEGMENT_COMPLETE', step=until, checkpoint=str(dest / f'checkpoint{until}.pt'), recipe_hash=recipe_hash(cfg), model_qualified=False))
        parallel.barrier()
    return str(dest / f'checkpoint{until}.pt')
