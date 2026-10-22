"""Focused retry-path test in fresh isolated namespaces, no solver changes."""
import copy
import json
import subprocess
import sys
from g1_pipeline.common import ROOT,write_json,read_jsonl,sha
from g1_pipeline.parallel_runner import initialize,retryable_io
from g1_pipeline.batch_state import State
from parallel_compare import compare


def prepare(name,bad_hash=False):
 directory=ROOT/'work'/name
 if directory.exists():raise ValueError('IO_TEST_NAMESPACE_EXISTS')
 task=copy.deepcopy(json.loads((ROOT/'audit/parallel_equivalence_panel.json').read_text())['tasks'][0])
 if bad_hash:task['mapping']['native_sha256']='0'*64
 panel=ROOT/'audit'/(name+'_panel.json');write_json(panel,{'scope':'isolated retry-path test, full original clip','tasks':[task]})
 directory.mkdir();design=initialize(directory,panel);write_json(directory/'run.json',design)
 state=State(directory,design);state.register([task['row']],'v25_parallel_full_clip_v1');state.close()
 return directory,panel

def row(directory):
 design=json.loads((directory/'run.json').read_text());state=State(directory,design)
 rows=state.rows();state.close();assert len(rows)==1;return rows[0]

def worker(directory,inject=False):
 code="from pathlib import Path\nfrom g1_pipeline import parallel_clip\nfrom g1_pipeline.parallel_runner import worker\n"
 if inject:
  code+="def transient(*args,**kwargs):\n    raise BlockingIOError(11,'controlled isolated benchmark transient I/O')\nparallel_clip.execute=transient\n"
 code+="worker(Path("+repr(str(directory))+"),0)\n"
 log=(directory/'test_processes.log').open('ab')
 proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 log.close();assert proc.returncode==0

def resume(directory,panel):
 log=(directory/'test_processes.log').open('ab')
 proc=subprocess.run([sys.executable,'-m','g1_pipeline.parallel_runner','run','--namespace',directory.name,'--panel',str(panel.relative_to(ROOT)),
                      '--workers','1','--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 log.close();assert proc.returncode==0

def run():
 once,p=prepare('benchmark_io_once_v4');worker(once,True);before=row(once)
 assert before['status']=='ERROR' and before['attempt']==1 and retryable_io(before)
 failed_output=json.loads(before['output_json']);failed_sha=sha(failed_output['entry_path'])
 resume(once,p);after=row(once)
 assert after['attempt']==2 and after['status']=='QA_PASS' and not retryable_io(after)
 assert sha(failed_output['entry_path'])==failed_sha and compare(once.name)['status']=='PASS'
 resume(once,p);assert row(once)==after
 exhausted,p=prepare('benchmark_io_exhausted_v4');worker(exhausted,True);worker(exhausted,True)
 terminal=row(exhausted);assert terminal['status']=='ERROR' and terminal['attempt']==2 and not retryable_io(terminal)
 worker(exhausted,True);assert row(exhausted)==terminal
 resume(exhausted,p);assert row(exhausted)==terminal and len(read_jsonl(exhausted/'results.jsonl'))==1
 schema,p=prepare('benchmark_no_retry_schema_v4',True);worker(schema)
 deterministic=row(schema);assert deterministic['status']=='ERROR' and deterministic['attempt']==1 and not retryable_io(deterministic)
 resume(schema,p);assert row(schema)==deterministic
 assert 'SEALED_HML_INPUT_CHANGED' in json.loads(open(json.loads(deterministic['output_json'])['entry_path']).read())['error']
 result={'status':'PASS','transient_io_retry_limit_total_attempts':2,'successful_retry_attempt':after['attempt'],
         'exhausted_error_attempts':terminal['attempt'],'deterministic_schema_error_attempts':deterministic['attempt'],
         'failed_attempt_artifacts_preserved':True,'successful_retry_matches_original_full_clip':True,
         'third_attempt_prevented':True,'committed_success_not_recomputed':True,
         'injection_scope':'RAM-only replacement of execute in newly spawned benchmark test processes; original source files untouched',
         'executor_sha256':sha(ROOT/'g1_pipeline/parallel_runner.py'),
         'namespaces':[d.name for d in [once,exhausted,schema]],'production_release':False}
 write_json(ROOT/'audit/parallel_io_retry_test.json',result);print('BOUNDED IO RETRY PASS')
if __name__=='__main__':run()
