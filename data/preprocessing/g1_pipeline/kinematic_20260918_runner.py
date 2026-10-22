"""Bound recovery of unchanged conversion; BONES admission is a separate overlay."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,types,importlib.abc
from .common import ROOT,sha
from .kinematic_policy import BASE,load

AUTH=BASE/'bones_official_20260918/recovery_binding.json'

def require_conversion_frozen():
    from .calibration_protocol import PATH
    from .gates import require_pass
    require_pass('00_preflight')
    policy=load(PATH)
    if policy['status']!='FROZEN_BEFORE_HELD_OUT':raise ValueError('INVALID_CALIBRATION_FREEZE')
    bound_check(policy['bindings'])
    return policy

def bound_check(bindings):
    from .parallel_runner import bound_check as original
    auth=load(AUTH)
    for path,bound in auth['bindings'].items():
        if sha(ROOT/path)!=bound:raise ValueError('RECOVERY_BINDING_CHANGED:'+path)
    changed=auth['unrelated_training_file'];updated=dict(bindings)
    if changed['path'] in updated:
        if updated[changed['path']]!=changed['original_sha256']:raise ValueError('UNKNOWN_PRIOR_TRAINING_BINDING')
        updated[changed['path']]=changed['current_sha256']
    original(updated)

class NoUnrelatedTrainingImport(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[-1]=='main3_runtime':raise ImportError('UNRELATED_TRAINING_IMPORT_FORBIDDEN_IN_CONVERSION')

def replace_code(code):
    constants=tuple(replace_code(c) if isinstance(c,types.CodeType) else
                    'g1_pipeline.kinematic_20260918_runner' if c=='g1_pipeline.kinematic_runner' else c for c in code.co_consts)
    return code.replace(co_consts=constants)

def compiled():
    from . import kinematic_runner
    ns=kinematic_runner.compiled()
    for value in list(ns.values()):
        if isinstance(value,types.FunctionType) and value.__globals__ is ns:value.__code__=replace_code(value.__code__)
    ns['bound_check']=bound_check
    def initialize(directory,panel):
        # Avoid patching the original module: compile the small initializer with a
        # local-import replacement that resolves the bound recovery check.
        import ast
        tree=ast.parse((ROOT/'g1_pipeline/kinematic_runner.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize')
        fn.body=[n for n in fn.body if not isinstance(n,ast.ImportFrom)]
        env=dict(vars(kinematic_runner),bound_check=bound_check)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'scoped_recovery_initializer','exec'),env)
        return env['initialize'](directory,panel)
    ns['initialize']=initialize
    retry=ns['retryable_io']
    def retry_preexecution(record):
        if retry(record):return True
        auth=load(AUTH)
        if record['sample_id']!=auth['retry_preexecution_sample'] or record['status']!='ERROR' or record['attempt']>=2:return False
        import json
        output=json.loads(record['output_json']);entry=load(__import__('pathlib').Path(output['entry_path']))
        return entry.get('error')==auth['retry_preexecution_error']
    ns['retryable_io']=retry_preexecution
    # Original compiled worker imports execute at call time. Replace only this import.
    # Existing worker already includes integrity guard; rewrite its import's name
    # in the compiled code object without changing any control flow.
    worker=ns['worker'];worker.__code__=worker.__code__.replace(co_consts=tuple('bones_official_admission' if c=='kinematic_clip' else c for c in worker.__code__.co_consts),co_names=tuple('bones_official_admission' if n=='kinematic_clip' else n for n in worker.__code__.co_names))
    return ns

if __name__=='__main__':
    sys.meta_path.insert(0,NoUnrelatedTrainingImport())
    bound_check({})
    compiled()['main']()
