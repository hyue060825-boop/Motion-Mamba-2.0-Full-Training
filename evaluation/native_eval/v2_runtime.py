"""One GPU, own lock, completion pipeline, persistent tmux entrypoint."""
from motion_mamba2.paths import artifact_path
import fcntl,os,subprocess,sys,time
from .common import *
def acquire():
    lock=open(ROOT/'gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines();used={v.strip() for v in apps}
        for line in cards.splitlines():
            index,uuid,memory,util=[v.strip() for v in line.split(',')]
            if int(index)>=4 and uuid not in used and int(memory)<100 and int(util)<5:
                os.environ['CUDA_VISIBLE_DEVICES']=uuid;os.environ['EVALUATOR_V2_GPU_UUID']=uuid
                write(ROOT/'GPU_LEASE.json',dict(pid=os.getpid(),index=int(index),uuid=uuid,acquired_at=time.time(),prior_charged_seconds=read(ROOT/'GPU_BUDGET.json')['seconds'],initial_gpu_snapshot=cards,initial_process_uuids=apps))
                ledger('gpu_acquired',uuid=uuid,index=int(index));return lock
        status('WAITING_FOR_IDLE_GPU',retry_seconds=900,reason='all cards 4-7 occupied; no other process interrupted');time.sleep(900)
def main():
    lock=acquire()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from .v2_train import train
    from .v2_panels import cache_missing_mirrors
    from .v2_validate import select,confirm
    from .v2_delivery import deliver
    try:
        cache_missing_mirrors()
        pilot=train('humanml3d','R2',1701,True);quotas={}
        for d in ('humanml3d','bones'):
            n=len({r['source_family_id'] for r in rows(ROOT/'data'/f'{d}_train.jsonl')});steps=(n+31)//32
            quotas[d]=dict(families=n,updates_max=50*steps,max_epochs=50,min_epochs=10,patience=8,eta_upper_seconds=pilot['seconds_per_update']*steps*50*2.)
        hours=sum(v['eta_upper_seconds'] for v in quotas.values())*5/3600
        if hours>21:raise RuntimeError('Pilot forecast exceeds reserved budget; revise bounded plan before training')
        plan=dict(frozen_at=time.time(),pilot=pilot,quotas=quotas,variants=['R0','R1','R2'],primary_seed=1701,sensitivity_seed=2903,total_train_upper_hours=hours,additional_budget_gpu_hours=24,disk_gib=50,second_seed_rule='only final selected variants; no seed cherry-pick',optimizer=read(CODE/'config.json'),selection='separate retrieval maximum and reconstruction minimum under retrieval/non-collapse constraints',v1_gpu_seconds=read(artifact_path('gpu_budget_json_44a0332f70'))['seconds'])
        path=ROOT/'TRAIN_PLAN_V2.json'
        if not (ROOT/'TRAIN_PLAN_FROZEN.json').exists():write(path,plan);write(ROOT/'TRAIN_PLAN_FROZEN.json',dict(sha256=sha(path)))
        print('PLAN',dict(estimated_train_upper_hours=hours),flush=True)
        for variant in ('R0','R1','R2'):
            for domain in ('humanml3d','bones'):train(domain,variant,1701)
        selected=select()
        for domain,heads in selected['selected'].items():
            for variant in sorted({v['variant'] for v in heads.values()}):train(domain,variant,2903)
        confirm();return deliver()
    except Exception as e:
        status('V2_FAILED_RESUMABLE',error=repr(e),pid=os.getpid());raise
    finally:
        lease=read(ROOT/'GPU_LEASE.json');lease['released_at']=time.time();lease['reserved_wall_seconds']=lease['released_at']-lease['acquired_at'];write(ROOT/'GPU_LEASE.json',lease)
        budget=read(ROOT/'GPU_BUDGET.json');extra=max(0.,lease['prior_charged_seconds']+lease['reserved_wall_seconds']-budget['seconds']);budget['seconds']+=extra
        if extra:budget['stages'].append(dict(stage='reserved_gpu_between_measured_stages',seconds=extra))
        write(ROOT/'GPU_BUDGET.json',budget);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
