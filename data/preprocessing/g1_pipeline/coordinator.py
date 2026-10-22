"""Ordered stage dispatch. Every downstream runtime checks the upstream gate."""
import importlib
from .gates import require_pass,audit
from .common import ROOT

STAGES=[('01_bones_calibration','00_preflight','calibration'),
        ('02_hml100','01_bones_calibration','production_hml'),
        ('03_hiphi100','02_hml100','production_hiphi'),
        ('04_fusion_qa','03_hiphi100','production_fusion'),
        ('05_full_batch','04_fusion_qa','production_batch'),
        ('06_training_data_ready','05_full_batch','production_training_readiness')]


def stage(stage_name,*,resume=False,dry_run=False,workers=1,max_clips=None,seed=20260915):
    spec=next(x for x in STAGES if x[0]==stage_name);name,previous,module=spec
    require_pass(previous)
    if workers!=1 or seed!=20260915:raise ValueError('FROZEN_INITIAL_RUN_REQUIRES_WORKERS_1_SEED_20260915')
    if name=='01_bones_calibration':
        from .calibration_protocol import freeze
        from .frozen_calibration_runner import run
        if dry_run:return {'stage':name,'fixed_acceptance':300,'development':200,'smoke':16,'upstream':'PASS','freeze_required_before_acceptance':True}
        freeze()
        if max_clips is not None:raise ValueError('FIXED_ACCEPTANCE_PANEL_CANNOT_BE_TRUNCATED')
        run('dev',resume=resume,verification=True)
        return run('accept',resume=resume)
    # Adding a later-stage module does not change upstream solver semantics.
    path=ROOT/'g1_pipeline'/(module+'.py')
    if not path.is_file():raise RuntimeError('STAGE_IMPLEMENTATION_PENDING:'+module)
    return importlib.import_module('g1_pipeline.'+module).run(resume=resume,dry_run=dry_run,workers=workers,max_clips=max_clips,seed=seed)


def pipeline(**kwargs):
    try:require_pass('00_preflight')
    except RuntimeError:
        if kwargs.get('dry_run'):return {'status':'BLOCKED','stage':'00_preflight','reason':'audit required'}
        if audit()['status']!='PASS':return {'status':'FAIL','stage':'00_preflight'}
    for name,previous,module in STAGES:
        try:require_pass(name)
        except RuntimeError:
            result=stage(name,**kwargs)
            if kwargs.get('dry_run'):return result
            require_pass(name)
    return {'status':'TRAINING_DATA_READY'}
