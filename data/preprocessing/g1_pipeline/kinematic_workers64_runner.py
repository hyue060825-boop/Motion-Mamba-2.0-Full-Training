"""Allow the user-authorized64-worker trial in the unchanged bounded scheduler."""
import sys
from .kinematic_20260918_runner import compiled as original,bound_check,NoUnrelatedTrainingImport


def compiled():
    ns=original();run=ns['run'];constants=list(run.__code__.co_consts)
    allowed=(1,2,4,16,32,48)
    if constants.count(allowed)!=1:raise ValueError('WORKER_LIMIT_PATCH_NOT_UNIQUE')
    constants[constants.index(allowed)]=allowed+(64,)
    run.__code__=run.__code__.replace(co_consts=tuple(constants))
    return ns


if __name__=='__main__':
    sys.meta_path.insert(0,NoUnrelatedTrainingImport());bound_check({})
    compiled()['main']()
