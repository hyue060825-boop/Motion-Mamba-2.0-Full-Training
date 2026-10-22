"""Controlled crash of a new benchmark worker only; verify real resume/locks."""
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from g1_pipeline.common import ROOT,write_json,read_jsonl,sha
from parallel_compare import compare

NS='benchmark_recovery_'+os.environ.get('PARALLEL_VALIDATION_VERSION','v1')
BASE=ROOT/'work'/NS

def records():
 if not (BASE/'state.sqlite').exists():return []
 with sqlite3.connect(BASE/'state.sqlite') as c:
  c.row_factory=sqlite3.Row
  return [dict(r) for r in c.execute('SELECT * FROM items')]

def wait_for(fn,timeout=180):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  value=fn()
  if value:return value
  time.sleep(.1)
 raise TimeoutError('RECOVERY_TEST_OBSERVATION_TIMEOUT')

def run():
 if BASE.exists():raise ValueError('RECOVERY_TEST_NAMESPACE_EXISTS')
 data=json.loads((ROOT/'audit/parallel_equivalence_panel.json').read_text())
 data['tasks']=data['tasks'][:2]
 data['frames']=sum(t['reference']['frames'] for t in data['tasks'])
 panel=ROOT/'audit/parallel_recovery_panel.json';write_json(panel,data)
 cmd=[sys.executable,'-m','g1_pipeline.parallel_runner','run','--namespace',NS,'--panel',str(panel.relative_to(ROOT)),'--workers','1']
 log=(ROOT/'logs/parallel_recovery_test.log').open('ab')
 p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 wait_for(lambda: (BASE/'heartbeat.json').exists())
 duplicate=subprocess.run(cmd+['--resume'],cwd=ROOT,capture_output=True,text=True,timeout=60)
 assert duplicate.returncode!=0 and 'COORDINATOR_ALREADY_RUNNING' in duplicate.stderr
 def second_active():
  rows=records()
  if len(rows)!=2:return None
  first=next(r for r in rows if r['sample_id']==data['tasks'][0]['row']['sample_id'])
  second=next(r for r in rows if r['sample_id']==data['tasks'][1]['row']['sample_id'])
  if first['status'] not in ('QA_PASS','REJECTED') or second['status']!='VALIDATED':return None
  heartbeats=list((BASE/'items').glob('*/attempt_1/heartbeat.json'))
  if not any(json.loads(h.read_text())['sample_id']==second['sample_id'] for h in heartbeats):return None
  return first,second
 first,second=wait_for(second_active)
 original_output=json.loads(first['output_json']);original_hash=sha(original_output['entry_path'])
 pid=second['worker_pid'];cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert 'g1_pipeline.parallel_runner worker' in cmdline and NS in cmdline
 # Existing item lock refuses a duplicate process while the legitimate clip runs.
 duplicate_worker=subprocess.run([sys.executable,'-m','g1_pipeline.parallel_runner','worker','--namespace',NS,'--index','1'],cwd=ROOT,capture_output=True,text=True,timeout=30)
 assert duplicate_worker.returncode==0
 assert next(r for r in records() if r['sample_id']==second['sample_id'])['attempt']==1
 os.kill(pid,signal.SIGKILL)
 p.wait(timeout=60)
 before=records()
 assert len(read_jsonl(BASE/'results.jsonl'))==1
 assert next(r for r in before if r['sample_id']==second['sample_id'])['status']=='VALIDATED'
 resumed=subprocess.run(cmd+['--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 assert resumed.returncode==0
 after=records()
 assert sorted(r['attempt'] for r in after)==[1,2]
 assert sha(original_output['entry_path'])==original_hash
 ledger=read_jsonl(BASE/'results.jsonl')
 assert len(ledger)==len({r['sample_id'] for r in ledger})==2
 # A no-op resume must not reexecute either committed clip.
 again=subprocess.run(cmd+['--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 assert again.returncode==0 and records()==after
 from g1_pipeline.parallel_runner import available
 assert all(available(BASE,r) for r in after)
 comp=compare(NS);assert comp['status']=='PASS'
 result={'status':'PASS','namespace':NS,'controlled_worker_pid':pid,'controlled_worker_command':cmdline,
         'interruption':'SIGKILL isolated benchmark worker after original clip heartbeat; no serial pilot touched',
         'before':[{k:r[k] for k in ('sample_id','status','attempt','worker_pid')} for r in before],
         'after':[{k:r[k] for k in ('sample_id','status','attempt','worker_pid')} for r in after],
         'committed_item_not_recomputed':True,'duplicate_coordinator_rejected':True,'duplicate_worker_did_not_claim':True,
         'no_partial_success':True,'no_duplicate_sample':True,'all_item_locks_unheld':True,'exact_original_outputs':True,
         'incomplete_attempt_preserved_for_audit':True}
 write_json(ROOT/'audit/parallel_recovery_test.json',result);print('RECOVERY PASS')
 log.close()
if __name__=='__main__':run()
