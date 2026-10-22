from motion_mamba2.paths import artifact_path
import os,sys,json,argparse,copy,time
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from motion_training import evaluation,acceleration
from motion_training.io import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);a=p.parse_args();folder=D/'evaluation'/str(a.step);folder.mkdir(parents=True,exist_ok=True)
# Reuse the exact prior Old/New evaluation panel and metrics, only candidate path changes.
cfg=read(artifact_path('old_json_9aee84d268'));cfg.update(parent=str(D/'train'/f'checkpoint{a.step}.pt'),budget_root=str(D/'budget'),text_roots=read(D/'configs/train.json')['text_roots'])
cfg['kernel_gate_path']=str(folder/'kernel/GATE.json');path=folder/'config.json';write(path,cfg)
try:acceleration.gate(str(path),str(folder/'kernel'))
except AssertionError as error:
 write(folder/'KERNEL_FALLBACK.json',dict(reason=repr(error),evaluation_backend='reference',training_changed=False))
for split in (['DEV','FINAL'] if a.step==100000 else ['DEV']):
 c=copy.deepcopy(cfg);c['evaluation_split']=split;c['evaluation_source_indices']=[10,11] if split=='FINAL' else [3,4];cf=folder/f'config_{split}.json';write(cf,c);r=evaluation.evaluate(cfg['parent'],str(folder/split),str(cf));print('EVAL_COMPLETE',a.step,split,r['foundation']['function'],r['foundation']['joint'],flush=True)
write(folder/'COMPLETE.json',dict(complete=True,checkpoint=cfg['parent'],checkpoint_sha256=sha(cfg['parent']),candidate_stage_step=a.step,old_reference=artifact_path('old_c1509953a8'),common_panel=True,physical_simulation_verified=False,do_not_deploy_without_actual_simulation=True))
