"""Small local process coordinator. Benchmark/pilot only; no production writes.

One fresh process per full clip, existing WAL/item locks, one ledger writer.
STOP or SIGTERM drains active clips. Orphaned live workers keep per-item locks;
resume waits for their commit instead of claiming based on elapsed time.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import ROOT, sha, digest, now, write_json, write_jsonl, read_jsonl
from .batch_state import State, TERMINAL


def load(path):
    return json.loads(Path(path).read_text())


def bound_check(bindings):
    for path, expected in bindings.items():
        if sha(ROOT/path) != expected:
            raise ValueError('PARALLEL_INPUT_CHANGED:' + path)


def initialize(directory, panel):
    from .pilot_scope import require_pilot
    from .calibration_protocol import require_frozen, PATH
    from . import locked_target
    tasks = load(panel)['tasks']
    if not tasks or len({t['row']['sample_id'] for t in tasks}) != len(tasks):
        raise ValueError('DUPLICATE_OR_EMPTY_PARALLEL_PANEL')
    for source in {t['source'] for t in tasks}:
        require_pilot(source)
    policy = require_frozen()
    bindings = dict(policy['bindings'])
    bindings[str(PATH.relative_to(ROOT))] = sha(PATH)
    bindings[str(panel.relative_to(ROOT))] = sha(panel)
    code = {}
    source_designs = {}
    for source in {t['source'] for t in tasks}:
        mod = __import__('g1_pipeline.production_'+('hml' if source=='humanml3d' else 'hiphi'), fromlist=['run'])
        design = mod.run(pilot=True, dry_run=True)
        source_designs[source] = design
        code.update(design['fingerprint']['code'])
    for name, value in code.items():
        bindings['g1_pipeline/'+name+'.py'] = value
    for name in ('parallel_runner','parallel_clip','batch_state','pilot_scope'):
        bindings['g1_pipeline/'+name+'.py'] = sha(ROOT/'g1_pipeline'/(name+'.py'))
    from .pilot_scope import AUTH
    bindings[str(AUTH.relative_to(ROOT))] = sha(AUTH)
    # Original prepare() runs unchanged with only its output destinations scoped.
    old_dir, old_write = locked_target.ASSET_DIR, locked_target.write_json
    locked_target.ASSET_DIR = directory/'asset'
    locked_target.write_json = lambda p, obj: write_json(directory/'asset_equivalence.json', obj)
    try:
        xml, report = locked_target.prepare()
    finally:
        locked_target.ASSET_DIR, locked_target.write_json = old_dir, old_write
    for p,h in report['source']['resolved_meshes'].items(): bindings[p] = h
    bindings[report['source']['xml']] = report['source']['xml_hash']
    asset = {'path':str(xml), 'sha256':sha(xml), 'report':report}
    scope = 'isolated_benchmark' if directory.name.startswith('benchmark_') else 'isolated_pilot'
    compatibility = {'execution_scope': scope}
    if scope == 'isolated_pilot':
        if len(source_designs) != 1:
            raise ValueError('PILOT_REQUIRES_SINGLE_SOURCE')
        original = next(iter(source_designs.values()))['fingerprint']
        fingerprint = {**original, 'code': dict(original['code'])}
        for name in ('parallel_runner', 'parallel_clip', 'batch_state', 'pilot_scope'):
            fingerprint['code'][name] = bindings['g1_pipeline/'+name+'.py']
        compatibility['fingerprint'] = fingerprint
    return {**compatibility, 'scope':scope,
            'namespace':directory.name,'panel':str(panel.relative_to(ROOT)), 'expected':len(tasks),
            'seed':20260915, 'threads':{n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            'bindings':bindings,'code':code,'asset':asset,'training_release':False,
            'state_QA_PASS_means':'artifact commit only; original clip status retains review pending; no acceptance gate written'}


def retryable_io(record):
    """At most two total attempts; schema/IK/QA failures are never retried."""
    if record['status']!='ERROR' or record['attempt']>=2 or not record.get('output_json'):
        return False
    output=json.loads(record['output_json'])
    entry=load(output['entry_path']);error=entry.get('error','')
    import re
    # Only explicit operating-system transient I/O errors. Invalid native input,
    # missing files, permission failures and numerical errors remain terminal.
    return bool(re.match(r'^(BlockingIOError|InterruptedError|TimeoutError)\(',error)
                or re.match(r'^OSError\((4|5|11|110|116),',error))


def worker(directory, index):
    # Only this isolated test entry can be terminated by the recovery harness.
    import resource
    from .parallel_clip import execute
    design = load(directory/'run.json'); tasks = load(ROOT/design['panel'])['tasks']
    task = tasks[index]; sid = task['row']['sample_id']
    state = State(directory, design)
    record=next(r for r in state.rows() if r['sample_id']==sid)
    with state.own(sid, resume=True, retry_failed=retryable_io(record)) as owned:
        if owned is None:
            state.close(); return
        attempt = next(r['attempt'] for r in state.rows() if r['sample_id']==sid)
        out = directory/'items'/digest(sid)[:20]/('attempt_'+str(attempt))
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'owner.json', {'pid':os.getpid(),'sample_id':sid,'attempt':attempt,'time':now()})
        start = time.monotonic()
        try:
            bound_check(design['bindings'])
            entry, metrics = execute(task, out, design['asset'], design['code'])
        except Exception as exc:
            entry = {'sample_id':sid,'dataset':task['source'],'status':'ERROR','error':repr(exc),
                     'traceback':__import__('traceback').format_exc(),'attempt':attempt}
            metrics = {}
        usage = resource.getrusage(resource.RUSAGE_SELF)
        entry['attempt'] = attempt
        metrics.update(wall_sec=time.monotonic()-start, cpu_sec=usage.ru_utime+usage.ru_stime,
                       peak_rss_bytes=usage.ru_maxrss*1024, read_blocks=usage.ru_inblock, write_blocks=usage.ru_oublock,
                       pid=os.getpid(), threads=design['threads'])
        write_json(out/'execution.json',metrics); write_json(out/'result.json',entry)
        artifacts = {str(ROOT/p):h for p,h in entry.get('output_hashes',{}).items()}
        artifacts.update({str(out/n):sha(out/n) for n in ('execution.json','result.json')})
        output = {'artifact_hashes':artifacts,'entry_path':str(out/'result.json'),
                  'execution_path':str(out/'execution.json'), 'frames':entry.get('frames',0)}
        output.update({k:str(ROOT/v) for k,v in entry.get('output_paths',{}).items()})
        terminal = 'ERROR' if entry['status']=='ERROR' else 'REJECTED' if entry['status']=='REJECTED' else 'QA_PASS'
        state.commit(sid, terminal, output, entry.get('error'))
    state.close()


def available(directory, record):
    # Kernel lock, never a heartbeat timeout, decides whether a task is live.
    with (directory/'locks'/(record['item_key']+'.lock')).open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def import_serial_pilot(directory, state, tasks, design, oldbase):
    """Reference verified serial artifacts in a new namespace, never rewrite them."""
    old_design=load(oldbase/'run.json')
    for name,bound in old_design['fingerprint']['code'].items():
        if design['code'].get(name)!=bound:raise ValueError('SERIAL_REUSE_ALGORITHM_CHANGED:'+name)
    source=tasks[0]['source']
    static=ROOT/('audit/hml_native_rest_mapping_v2.json' if source=='humanml3d' else 'audit/hiphi_native_rest_mapping_v1.json')
    if sha(static)!=old_design['fingerprint']['static_mappings']:raise ValueError('SERIAL_REUSE_STATIC_CHANGED')
    if source=='hiphi' and sha(ROOT/'audit/hiphi_package_evidence_v1.json')!=old_design['fingerprint']['package_evidence']:
        raise ValueError('SERIAL_REUSE_PACKAGE_CHANGED')
    prior=read_jsonl(oldbase/'results.jsonl') if (oldbase/'results.jsonl').exists() else []
    if len(prior)!=len({r['sample_id'] for r in prior}):raise ValueError('SERIAL_LEDGER_DUPLICATES')
    byid={t['row']['sample_id']:t for t in tasks}
    if not {r['sample_id'] for r in prior}<=set(byid):raise ValueError('SERIAL_LEDGER_OUTSIDE_PANEL')
    from .common import HML
    imported=[]
    for entry in prior:
        sid=entry['sample_id'];task=byid[sid];mp=task['mapping']
        if entry['status'] not in ('KINEMATIC_CANDIDATE','REVIEW_REQUIRED','REJECTED') or not entry.get('output_hashes'):
            raise ValueError('SERIAL_REUSE_UNVERIFIABLE_TERMINAL:'+sid)
        bound_check(entry['output_hashes'])
        native=HML/task['row']['source_paths']['native'] if source=='humanml3d' else ROOT/mp['native_path']
        if sha(native)!=mp['native_sha256'] or sha(native)!=entry['source_sha256']:raise ValueError('SERIAL_REUSE_NATIVE_CHANGED')
        if sha(ROOT/mp['config_path'])!=mp['config_sha256'] or entry['mapping_sha256']!=mp['config_sha256']:
            raise ValueError('SERIAL_REUSE_MAPPING_CHANGED')
        if sha(ROOT/mp['audit_path'])!=mp['audit_sha256']:raise ValueError('SERIAL_REUSE_MAPPING_AUDIT_CHANGED')
        if source=='humanml3d':
            provenance=load(ROOT/mp['audit_path'])['source_provenance']
            for key,name in [('native_model','native_model_sha256'),('dmpl_model','dmpl_model_sha256')]:
                path=provenance.get(key) or provenance['native_model'].replace('/smplh/','/dmpls/')
                if sha(path)!=provenance[name]:raise ValueError('SERIAL_REUSE_BODY_MODEL_CHANGED')
        contract=load(ROOT/entry['output_paths']['contract_path'])
        if contract['asset_hash']!=design['asset']['report']['source']['asset_hash']:raise ValueError('SERIAL_REUSE_TARGET_CHANGED')
        with state.own(sid,resume=True) as owned:
            if owned is None:continue
            out=directory/'imports'/digest(sid)[:20];out.mkdir(parents=True,exist_ok=True)
            write_json(out/'result.json',entry)
            write_json(out/'execution.json',{'reused':True,'original_namespace':oldbase.name,'original_attempt':entry['attempt']})
            artifacts={str(ROOT/p):h for p,h in entry['output_hashes'].items()}
            artifacts.update({str(out/n):sha(out/n) for n in ('result.json','execution.json')})
            output={'artifact_hashes':artifacts,'entry_path':str(out/'result.json'),'execution_path':str(out/'execution.json'),
                    'frames':entry['frames'],**{k:str(ROOT/v) for k,v in entry['output_paths'].items()}}
            state.commit(sid,'REJECTED' if entry['status']=='REJECTED' else 'QA_PASS',output)
            imported.append(sid)
    write_json(directory/'serial_reuse.json',{'time':now(),'original_namespace':oldbase.name,'verified_terminal':len(prior),
        'newly_imported':imported,'original_run_sha256':sha(oldbase/'run.json'),
        'original_ledger_sha256':sha(oldbase/'results.jsonl') if prior else None,
        'source_code_config_assets_outputs_verified':True,'original_production_and_pilot_gates_unchanged':True,
        'training_release':False})


def run(directory, panel, workers, resume=False, reverse=False):
    if not directory.name.startswith(('benchmark_', 'pilot_parallel_')):
        raise ValueError('BENCHMARK_OR_PILOT_NAMESPACE_REQUIRED')
    if workers not in (1,2,4,16,32,48): raise ValueError('UNSUPPORTED_WORKER_COUNT')
    run_started=time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'coordinator.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('COORDINATOR_ALREADY_RUNNING')
        if (directory/'run.json').exists():
            if not resume: raise ValueError('RESUME_REQUIRED')
            design=load(directory/'run.json'); bound_check(design['bindings'])
            if str(panel.relative_to(ROOT))!=design['panel']: raise ValueError('PANEL_CHANGED')
        else:
            design=initialize(directory,panel); write_json(directory/'run.json',design)
        tasks=load(panel)['tasks']
        if resume and (directory/'STOP').exists():
            (directory/'STOP').rename(directory/('STOP_consumed_'+str(time.time_ns())+'.json'))
        if design['scope']=='isolated_pilot':
            if len(tasks)!=100 or len({t['source'] for t in tasks})!=1:
                raise ValueError('PILOT_REQUIRES_ORIGINAL_FIXED100')
            # Independent pilot execution cannot overlap the old serial runner.
            source=tasks[0]['source']; short='hml' if source=='humanml3d' else 'hiphi'
            oldns=load(ROOT/f'audit/current_pilot_{short}100_run.json')['namespace']
            serial_lock=(ROOT/'work'/oldns/'run.lock').open('a+')
            try: fcntl.flock(serial_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('SERIAL_PILOT_STILL_RUNNING')
            original=read_jsonl(ROOT/('manifests/hml_smoke100.v2.jsonl' if short=='hml' else 'manifests/hiphi_smoke100.jsonl'))
            if [t['row'] for t in tasks]!=original: raise ValueError('PILOT_PANEL_CHANGED')
        state=State(directory,design); state.register([t['row'] for t in tasks],'v25_parallel_full_clip_v1')
        if design['scope']=='isolated_pilot':
            import_serial_pilot(directory,state,tasks,design,ROOT/'work'/oldns)
        # Verify terminal hashes once at the resume boundary, not every poll.
        for r in state.rows():
            if r['status'] in TERMINAL and not state.verify(r): raise ValueError('COMMITTED_ARTIFACT_CHANGED:'+r['sample_id'])
        initial_complete={r['sample_id'] for r in state.rows() if r['status'] in TERMINAL and not retryable_io(r)}
        order=list(range(len(tasks))); order=order[::-1] if reverse else order
        indexes={t['row']['sample_id']:i for i,t in enumerate(tasks)}
        stopping=False
        def stop(signum,frame):
            nonlocal stopping
            stopping=True
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        active={}; finished=set(); started=time.monotonic(); peak_rss=0; samples=[]; last_sample=0
        import psutil
        logdir=directory/'logs'; logdir.mkdir(exist_ok=True)
        while True:
            for index,(proc,log) in list(active.items()):
                if proc.poll() is not None:
                    log.close(); del active[index]
                    if proc.returncode:
                        # Do not retry deterministic errors or crashes silently.
                        stopping=True
                        write_json(directory/'worker_failure.json',{'index':index,'returncode':proc.returncode,'time':now()})
            rows=state.rows(); byid={r['sample_id']:r for r in rows}
            completed={r['sample_id'] for r in rows if r['status'] in TERMINAL and not retryable_io(r)}
            if completed!=finished:
                entries=[load(json.loads(byid[t['row']['sample_id']]['output_json'])['entry_path']) for t in tasks if t['row']['sample_id'] in completed]
                write_jsonl(directory/'results.jsonl',entries); finished=completed
            stopping=stopping or (directory/'STOP').exists()
            live_external=False
            if not stopping:
                for i in order:
                    if len(active)>=workers: break
                    sid=tasks[i]['row']['sample_id']; record=byid[sid]
                    if sid in completed or i in active: continue
                    if not available(directory,record): live_external=True; continue
                    log=(logdir/(str(i)+'.log')).open('ab')
                    proc=subprocess.Popen([sys.executable,'-m','g1_pipeline.parallel_runner','worker','--namespace',directory.name,'--index',str(i)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                    active[i]=(proc,log)
            if time.monotonic()-last_sample>=5:
                rss=0; cpu=0
                for proc,_ in active.values():
                    try:
                        p=psutil.Process(proc.pid); rss+=p.memory_info().rss; ct=p.cpu_times();cpu+=ct.user+ct.system
                    except psutil.Error: pass
                peak_rss=max(peak_rss,rss)
                samples.append({'elapsed_sec':time.monotonic()-started,'active':len(active),'rss_bytes':rss,'active_cpu_seconds':cpu,'load1':os.getloadavg()[0]})
                write_json(directory/'heartbeat.json',{'time':now(),'pid':os.getpid(),'completed':len(completed),'expected':len(tasks),'active_pids':[p.pid for p,_ in active.values()],'stopping':stopping})
                last_sample=time.monotonic()
            if not active and (stopping or len(completed)==len(tasks)): break
            time.sleep(.25)
        entries=read_jsonl(directory/'results.jsonl') if (directory/'results.jsonl').exists() else []
        seconds=time.monotonic()-run_started
        frames=sum(r.get('frames',0) for r in entries)
        newframes=sum(r.get('frames',0) for r in entries if r['sample_id'] not in initial_complete)
        throughput_valid = not resume and not initial_complete
        report={'time':now(),'status':'COMPLETE' if len(entries)==len(tasks) else 'STOPPED_AT_CLIP_BOUNDARY',
                'workers':workers,'expected':len(tasks),'completed':len(entries),'pending':len(tasks)-len(entries),
                'counts':dict(Counter(r['status'] for r in entries)),'wall_sec':seconds,'frames':frames,
                'newly_processed_frames':newframes,'reused_clips':len(initial_complete),
                'frames_per_sec':newframes/seconds if throughput_valid else None,
                'source_motion_hours_per_hour':newframes/25/seconds if throughput_valid else None,
                'throughput_valid':throughput_valid,'throughput_note':'Only fresh runs without imported or previously committed clips are valid throughput measurements',
                'render_included':False,'peak_sampled_worker_rss_bytes':peak_rss,'samples':samples,'reverse_dispatch':reverse,
                'training_release':False}
        write_json(directory/('session_'+str(time.time_ns())+'.json'),report)
        write_json(directory/'heartbeat.json',{'time':now(),'pid':os.getpid(),'completed':len(entries),'expected':len(tasks),'active_pids':[],'stopping':stopping,'phase':report['status']})
        write_json(directory/'summary.json',report);state.close()
        print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker','stop','status'])
    p.add_argument('--namespace',required=True);p.add_argument('--panel');p.add_argument('--workers',type=int,default=16)
    p.add_argument('--resume',action='store_true');p.add_argument('--reverse',action='store_true');p.add_argument('--index',type=int)
    a=p.parse_args()
    if '/' in a.namespace or not a.namespace.startswith(('benchmark_','pilot_parallel_')):raise ValueError('INVALID_NAMESPACE')
    directory=ROOT/'work'/a.namespace
    if a.command=='run': run(directory,(ROOT/a.panel).resolve(),a.workers,a.resume,a.reverse)
    elif a.command=='worker': worker(directory,a.index)
    elif a.command=='stop': write_json(directory/'STOP',{'time':now(),'request':'drain active clips'})
    else:
        print(json.dumps({'heartbeat':load(directory/'heartbeat.json') if (directory/'heartbeat.json').exists() else None,
                          'summary':{k:v for k,v in load(directory/'summary.json').items() if k!='samples'} if (directory/'summary.json').exists() else None}))

if __name__=='__main__': main()
