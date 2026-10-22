#!/usr/bin/env python3
"""G1-only ordered data pipeline. Formal training and deployment are excluded."""
import argparse
from g1_pipeline.gates import audit
from g1_pipeline.coordinator import stage,pipeline


def main():
    commands={'run_bones_calibration':'01_bones_calibration','run_hml_smoke100':'02_hml100',
        'run_hiphi_smoke100':'03_hiphi100','run_three_source_qa':'04_fusion_qa','run_full_batch':'05_full_batch'}
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=[*commands,'audit_environment','audit_contracts','build_lineage_and_splits','compute_train_stats','validate_training_release','run_pipeline'])
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-clips',type=int);p.add_argument('--workers',type=int,default=1);p.add_argument('--seed',type=int,default=20260915)
    a=p.parse_args();options={k:getattr(a,k) for k in ('resume','dry_run','max_clips','workers','seed')}
    try:
        if a.command=='audit_environment':
            from g1_pipeline.audit import environment
            result=environment()
        elif a.command=='audit_contracts':result=audit()
        elif a.command=='build_lineage_and_splits':
            from g1_pipeline.lineage import build_all
            result=build_all()
        elif a.command=='run_pipeline':result=pipeline(**options)
        elif a.command=='validate_training_release':
            from g1_pipeline.release_validation import run
            result=run()
        elif a.command=='compute_train_stats':
            from g1_pipeline.gates import require_pass
            require_pass('05_full_batch')
            from g1_pipeline.production_stats import run
            result=run(**options)
        else:result=stage(commands[a.command],**options)
        print(result)
    except RuntimeError as exc:
        print({'status':'BLOCKED','reason':str(exc)});raise SystemExit(2)


if __name__=='__main__':main()
