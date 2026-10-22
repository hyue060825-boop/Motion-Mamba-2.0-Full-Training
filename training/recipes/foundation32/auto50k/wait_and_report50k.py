"""No GPU work: wait for supervisor-owned50k evaluation, then aggregate."""
from motion_mamba2.paths import artifact_path
import fcntl,json,os,time,traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path
from report50k import evaluate_report,atomic,read
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
E=D/'evaluation/50000';OUT=E/'comparison';OUT.mkdir(parents=True,exist_ok=True)
lock=(D/'auto50k/watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def state(stage,**extra):
 atomic(D/'auto50k/STATUS.json',dict(stage=stage,pid=os.getpid(),time_beijing=now(),report=str(OUT/'REPORT.md'),**extra))
 if stage.startswith('WAIT'):
  (OUT/'REPORT.md').write_text('# 新旧50k自动测评：等待结果\n\n状态：'+stage+'\n\n更新时间（北京时间）：'+now()+'\n\n训练达到50k后，由现有监督程序完成新50k推理等价核验、基础96项和开发／保留各16条连续任务；本程序随后核对旧50k原始结果，生成对照及平均滑步率。此文件目前没有新50k质量结论。\n')
 print(stage,extra,flush=True)
try:
 if (OUT/'DONE.json').exists() and read(OUT/'DONE.json').get('complete'):
  state('ALREADY_COMPLETE');raise SystemExit(0)
 deadline=time.time()+24*3600;previous=None
 while True:
  assert time.time()<deadline,'24h wait budget exhausted'
  files=[E/'COMPLETE.json',E/'DEV/RESULT.json',E/'FINAL/RESULT.json',E/'LOAD_evaluate.json']
  if all(p.exists() for p in files) and all(read(p).get('complete',False) for p in files[:3]):break
  run=read(D/'RUN_STATE.json')
  assert run.get('stage')!='STOPPED_ERROR',run
  leader=run.get('supervisor_pid');assert leader,'Missing supervisor PID'
  cmd=Path('/proc')/str(leader)/'cmdline'
  assert cmd.exists() and str(D/'supervisor4.py') in cmd.read_bytes().decode(errors='replace'),'Training supervisor is no longer active'
  stage='WAIT_EVALUATION' if (D/'train/checkpoint50000.pt').exists() else 'WAIT_CHECKPOINT'
  if stage!=previous:state(stage,supervisor_pid=leader);previous=stage
  time.sleep(60)
 state('BUILD_REPORT')
 for attempt in range(1,4):
  try:
   result=evaluate_report(E,E/'LOAD_evaluate.json',OUT)
   state('COMPLETE',report_ready=True,delta=result['delta_new_minus_old']);break
  except Exception:
   if attempt==3:raise
   time.sleep(10)
except Exception as error:
 state('FAILED',error=repr(error),traceback=traceback.format_exc())
 (OUT/'REPORT.md').write_text('# 新旧50k自动测评：未完成\n\n时间（北京时间）：'+now()+'\n\n原因：'+repr(error)+'\n\n未生成可宣称成功的对照。详情见服务器auto50k/STATUS.json；原始训练和评测日志保留。\n')
 raise
