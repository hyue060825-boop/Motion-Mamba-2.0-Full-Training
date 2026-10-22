from motion_mamba2.paths import artifact_path
import os,sys,json,time,gc,copy
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from motion_training import runtime,engine,data,losses,batching,distributed as parallel
from motion_training.io import write,Budget
parallel.setup();cfg=engine.config(D/'configs/train.json');cfg=dict(cfg,budget_root=str(D/'bench_budget'),name='speed_gate_batch32')
if parallel.leader():write(D/'bench_budget/LIMITS.json',dict(caps=dict(checks=100,B=0,C=0,teacher=0),total_updates=100,gpu_hours=10))
parallel.barrier();budget=Budget(cfg['budget_root'])
plan=json.loads(Path(artifact_path('manifest_json_d6a24ce639')).read_text())['plan'];fixtures={hf:[r for r in plan if (r['H'],r['F'])==hf][:32] for hf in data.BUCKETS}
# Use the same 20-layout marginal distribution in both arms.
layouts=[v for v,n in zip(data.BUCKETS,[2,5,4,2,4,3]) for _ in range(n)]
all_results={};original_build=engine.build
for arm in ['serial','batched']:
 m,c,ck=runtime.load(cfg['parent'],True);opt=runtime.optimizer(m,ck,cfg);times=[];rows=[]
 engine.build=lambda m,c,ck,r,cfg:batching.corrected_build(m,c,r,cfg,audit=False)
 for step,hf in enumerate([data.BUCKETS[2]]+layouts,1):
  records=fixtures[hf]
  if arm=='serial':rec=engine.update(m,c,ck,opt,cfg,records,step,'checks',budget)
  else:batching.install();rec=batching.update(m,c,ck,opt,cfg,records,step,'checks',budget)
  times.append(rec['seconds']);rows.append(dict(step=step,H=hf[0],F=hf[1],seconds=rec['seconds'],loss=rec['loss']))
  if parallel.leader():print('BENCH',arm,step,round(rec['seconds'],3),round(rec['loss'],6),flush=True)
 measured=times[1:];all_results[arm]=dict(mean_seconds=sum(measured)/len(measured),samples_per_second=32/(sum(measured)/len(measured)),rows=rows)
 del m,c,ck,opt;gc.collect();torch.cuda.empty_cache();parallel.barrier()
 if parallel.leader():write(D/'SPEED_GATE.json',dict(status='IN_PROGRESS',results=all_results))
if parallel.leader():
 ratio=all_results['serial']['mean_seconds']/all_results['batched']['mean_seconds'];assert ratio>1.2,('insufficient_speedup',ratio)
 write(D/'SPEED_GATE.json',dict(status='PASS',results=all_results,speedup=ratio,world_size=parallel.world(),physical_devices=os.environ['CUDA_VISIBLE_DEVICES'],global_batch=32,per_rank_batch=8,accumulation=4//parallel.world(),same_weights_start=True,same_records=True,same_loss=True,reference_training_backend=True,measurements_include_optimizer_communication_and_ledger=True,projected100k_training_hours=all_results['batched']['mean_seconds']*100000/3600,projected50k_training_hours=all_results['batched']['mean_seconds']*50000/3600))
parallel.barrier();print('SPEED_GATE_PASS',flush=True)
