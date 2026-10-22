"""Extend the tested continuation to64; clip children keep their original entry."""
import argparse,types
from kinematic_workers48_continue import compiled as original


def compiled():
    ns=original()
    def replace(code):
        values=[]
        for value in code.co_consts:
            if isinstance(value,types.CodeType):value=replace(value)
            elif value==(16,32,48):value=(16,32,48,64)
            elif value=='g1_pipeline.kinematic_20260918_runner':value='g1_pipeline.kinematic_workers64_runner'
            elif value=='scripts/kinematic_workers48_continue.py':value='scripts/kinematic_workers64_continue.py'
            values.append(value)
        return code.replace(co_consts=tuple(values))
    for name in ('compiled_run','run'):ns[name].__code__=replace(ns[name].__code__)
    return ns


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=[16,32,48,64],required=True)
    compiled()['run'](p.parse_args().workers)
