"""Run an additional authorized checkpoint with the unchanged easy-v3 engine."""
import argparse,traceback,time
import execute
ADDED=['native110','native117','native122']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=ADDED);p.add_argument('--task',required=True);a=p.parse_args()
 execute.MODELS=ADDED
 try:execute.worker(a.model,a.task)
 except BaseException:
  execute.wr(execute.R/'models'/a.model/a.task/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
