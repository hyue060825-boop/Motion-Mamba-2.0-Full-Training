"""Only v2 entrypoints; no legacy pipeline dispatch."""
import argparse,json,os,sys
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
def main():
    p=argparse.ArgumentParser(description='Independent G1 native evaluator v2');sub=p.add_subparsers(dest='command',required=True)
    for name in ('audit','prepare','train-all','validate','freeze','report'):sub.add_parser(name)
    t=sub.add_parser('train');t.add_argument('--domain',choices=['humanml3d','bones'],required=True);t.add_argument('--variant',choices=['R0','R1','R2'],required=True);t.add_argument('--seed',type=int,required=True);t.add_argument('--pilot',action='store_true')
    s=sub.add_parser('score');s.add_argument('--release',required=True);s.add_argument('--samples',required=True);s.add_argument('--output',required=True);s.add_argument('--metric',choices=['fid','retrieval','both','event_order','full_sequence'],default='both');s.add_argument('--diagnostic',action='store_true');s.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.command in ('audit','prepare'):
        from .v2_prepare import prepare
        from .v2_panels import build
        result=dict(data=prepare(),panels=build())
    elif a.command=='train':
        from .v2_runtime import acquire
        lock=acquire()
        from .v2_train import train
        result=train(a.domain,a.variant,a.seed,a.pilot)
        lock.close()
    elif a.command=='train-all':
        from .v2_runtime import main as run
        result=run()
    elif a.command=='validate':
        from .v2_validate import confirm
        result=confirm()
    elif a.command in ('freeze','report'):
        from .v2_delivery import deliver
        result=deliver()
    else:
        from .v2_score import score
        result=score(a.release,a.samples,a.output,a.metric,a.diagnostic,a.device)
    print(json.dumps(result,ensure_ascii=False,default=str))
if __name__=='__main__':main()
