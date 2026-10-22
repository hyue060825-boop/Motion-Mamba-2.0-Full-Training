"""Freeze GT-only real banks and anchored candidate pools before any scoring."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,re,json,time,collections,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from execute import R,rd,wr,sha,digest
import numpy as np,torch
REL=Path(artifact_path('humanml3d-g1-native-v2_65063e489a'))
HEADROOT=R/'evaluator_assets'
HEADS={1701:REL,2903:Path(artifact_path('seed2903_b7f176163d'))}
sys.path.insert(0,str(REL/'code'))
sys.path.insert(0,artifact_path('_runtime_924bb4d0b9'))
from native_eval.v2_score import model_for,verify_release
from native_eval.encoding import encode_arrays
from motion_mamba2.skeleton import G1Skeleton34
from motion_mamba2.representation.reps.motion_representation import MotionMamba2MotionRep
def representation():
 sk=G1Skeleton34(load=False);sk.register_buffer('neutral_joints',torch.tensor(rd(REL/'SKELETON.json')['data'],dtype=torch.float32),persistent=False)
 return MotionMamba2MotionRep(sk,25),sk
def canonical(rep,arr):
 x=torch.tensor(np.stack(arr));lens=torch.full((len(x),),x.shape[1],dtype=torch.long)
 with torch.inference_mode():
  g=rep.inverse(x,is_normalized=False)
  y=rep(g['local_rot_mats'],g['root_positions'],to_normalize=False,to_canonicalize=False,lengths=lens)
 assert torch.equal(y[:,:,:3],x[:,:,:3])
 return y.numpy(),g

def verify_assets():
 bindings=rd(HEADROOT/'ASSET_BINDINGS.json')
 for p,h in bindings['absolute_asset_hashes'].items():assert sha(p)==h,('changed_HumanML3D_evaluator_asset',p)
 assert sha(HEADROOT/'retrieval_stats.npz')==sha(REL/'stats.npz')
 return bindings
