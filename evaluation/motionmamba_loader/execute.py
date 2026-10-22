"""Add only verified100k MotionMamba to each frozen parent cohort."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,argparse,importlib.util
R=Path(__file__).resolve().parent
P=Path(json.loads((R/'ADDON_PLAN.json').read_text())['parent_root'])
S=Path(artifact_path('g1_motionmamba_matched100k_20261022_3b602cb1f7'));TRAIN=S/'MotionMamba_G1'
spec=importlib.util.spec_from_file_location('frozen_parent_execute',P/'execute.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.R=R;p.MODELS=['motionmamba'];MODELS=p.MODELS
rd,wr,sha,digest,schedule=p.rd,p.wr,p.sha,p.digest,p.schedule
old_verify=p.verify

def verify():
 old_verify()
 bindings=rd(R/'FROZEN.json')
 for file,h in bindings['parent_sources'].items():assert sha(P/file)==h,('changed_parent_source',file)
 assert rd(TRAIN/'FINAL_SAVE_VERIFIED.json')['passed']
 assert rd(R/'CHECKPOINTS.json')['motionmamba']['sha256']==rd(TRAIN/'FINAL_SAVE_VERIFIED.json')['checkpoint_sha256']

def setup(method,out):
 assert method=='motionmamba'
 os.environ.update(BASELINE_RUN_ROOT=str(TRAIN),EVAL_MODEL=method,EVAL_LABEL=method,EVAL_ARCH=method,
  HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MOTION_MAMBA2_SEQUENCE_MODEL='transformer',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 sys.path[:0]=[str(S/'code/src/motion_training/_runtime'),str(S/'code/src'),str(S/'scripts'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04'),str(S/'deps')]
 import torch,gc
 from motion_training import runtime
 from motion_training._compat import legacy_loss
 from external_model import ExternalDenoiser
 from committed_rolling import CommittedMOTION_MAMBA2
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 info=rd(R/'CHECKPOINTS.json')[method];ck=torch.load(info['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
 assert ck['method']=='motionmamba' and ck['step']==ck['scheduler']['completed']==100000
 assert ck['exposures']==15200000 and len(ck['rng_by_rank'])==4 and len(set(ck['rank_state_hashes']))==1
 for file,h in ck['source_hashes'].items():assert sha(S/file)==h,('changed_training_source',file)
 m=runtime.load_model('g1',device='cuda:0',text_encoder=False,fresh_denoiser=True);legacy_loss._replace_stats(m,info['stats_dir'])
 raw=ExternalDenoiser(method,S/'repos/MotionMamba',m.motion_rep,m.hybrid).cuda();raw.load_state_dict(ck['model'],strict=True)
 assert all(torch.equal(v.detach().cpu(),ck['model'][k]) for k,v in raw.state_dict().items())
 m.denoiser.model=raw;m.generation_contract=ck['config']['generation_contract'];m.training_codec_contract=ck['config']['training_codec_contract']
 assert m.generation_contract=='g1_committed_v3' and m.decoder_history_frames==52
 m.eval();m.requires_grad_(False);out.mkdir(parents=True,exist_ok=True)
 wr(out/'LOAD.json',dict(all_loaded_denoiser_weights_verified=True,actual_step=100000,checkpoint_sha256=info['sha256'],
  method='motionmamba',architecture='official MotionMamba HTM/BSM common-G1 fixed256 adaptation',parameters=sum(x.numel() for x in raw.parameters()),
  source_adaptations=raw.source_edits,local_history_frames=196,accepted_window_frames=52,Native_memory_module=False,
  no_goal_cropping=True,no_future_GT_trajectory=True,weights_or_training_changed=False))
 gc.collect();torch.cuda.empty_cache();proxy=CommittedMOTION_MAMBA2(m,out/'INFERENCE_TRACE.json')
 return proxy,m,proxy
p.setup=setup;p.verify=verify

def worker(method,task):return p.worker(method,task)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',choices=MODELS,required=True);a.add_argument('--task',required=True);args=a.parse_args()
 try:worker(args.model,args.task)
 except BaseException:
  import traceback;wr(R/'models'/args.model/args.task/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
