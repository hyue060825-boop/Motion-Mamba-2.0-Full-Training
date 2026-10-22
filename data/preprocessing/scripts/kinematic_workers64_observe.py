"""Observe a64-worker live window, including clips across a batch boundary."""
import time,json,sqlite3,collections,psutil
from pathlib import Path
from g1_pipeline.common import ROOT,write_json,now
from g1_pipeline.kinematic_policy import BASE,load


def run():
    trial=BASE/'workers64_trial';launch=load(trial/'launch.json')
    while time.time()<launch['epoch']+100:time.sleep(5)
    start=time.time();samples=[]
    while time.time()-start<300:
        state=load(BASE/'status.json');rss=0;workers=[]
        try:workers=psutil.Process(state['coordinator_pid']).children()
        except psutil.NoSuchProcess:pass
        for worker in workers:
            try:rss+=worker.memory_info().rss
            except psutil.NoSuchProcess:pass
        samples.append({'epoch':time.time(),'active_workers':len(workers),'worker_rss_bytes':rss,
                        'memory_available_bytes':psutil.virtual_memory().available,
                        'cpu_percent':psutil.cpu_percent(interval=1)})
        time.sleep(9)
    end=time.time();state=load(BASE/'status.json');rows=[];checks=load(trial/'pre_resume_commits.json')['records']
    for index in range(int(launch['namespace'].rsplit('_',1)[1]),state['batch_index']+1):
        directory=ROOT/'work'/f'kinematic_training_20260917_v1_batch_{index:04d}'
        db=sqlite3.connect('file:'+str(directory/'state.sqlite')+'?mode=ro',uri=True)
        if directory.name==launch['namespace']:
            current={sid:(status,attempt,out) for sid,status,attempt,out in db.execute('select sample_id,status,attempt,output_json from items')}
            for r in checks:assert current[r['sample_id']]==(r['status'],r['attempt'],r['output_json']),'PRIOR_COMMIT_CHANGED'
        for inp,out in db.execute('select input_json,output_json from items where output_json is not null'):
            source=json.loads(inp);output=json.loads(out);path=Path(output['execution_path'])
            if not start<=path.stat().st_mtime<=end:continue
            execution=load(path);entry=load(Path(output['entry_path']))
            if entry.get('reused') or execution.get('reused'):continue
            contract=load(ROOT/entry['output_paths']['contract_path']) if entry.get('output_paths') else {}
            if entry.get('decision')=='ACCEPTED_KINEMATIC':assert contract['status']=='PASS' and contract['finite']
            rows.append({'sample_id':entry['sample_id'],'source':entry['dataset'],'category':source['category'],
                         'frames':entry.get('frames',0),'estimated_target_frames':source['estimated_target_frames'],
                         'wall_sec':execution['wall_sec'],'status':entry['status'],'decision':entry.get('decision'),
                         'error':entry.get('error'),'contract_status':contract.get('status'),'completed_epoch':path.stat().st_mtime})
        db.close()
    seconds=end-start;baseline=load(BASE/'workers48_trial/observation48.json')
    result={'time':now(),'workers':64,'window_start_epoch':start,'window_end_epoch':end,'window_seconds':seconds,
            'completed':len(rows),'frames':sum(r['frames'] for r in rows),'frames_per_sec':sum(r['frames'] for r in rows)/seconds,
            'source_counts':dict(collections.Counter(r['source'] for r in rows)),
            'status_counts':dict(collections.Counter(r['status'] for r in rows)),
            'baseline48_frames_per_sec':baseline['frames_per_sec'],'prior_commits_unchanged':True,
            'sampled_peak_worker_rss_bytes':max(r['worker_rss_bytes'] for r in samples),
            'sampled_min_available_memory_bytes':min(r['memory_available_bytes'] for r in samples),
            'mean_active_workers':sum(r['active_workers'] for r in samples)/len(samples),
            'mean_host_cpu_percent':sum(r['cpu_percent'] for r in samples)/len(samples),'samples':samples,'rows':rows,
            'limitation':'Adjacent production-queue windows, not identical clips; source/category effects and any batch-boundary idle time are retained. No reruns.'}
    result['observed_throughput_ratio']=result['frames_per_sec']/baseline['frames_per_sec']
    write_json(trial/'observation64.json',result);print({k:v for k,v in result.items() if k not in ('samples','rows')},flush=True)


if __name__=='__main__':run()
