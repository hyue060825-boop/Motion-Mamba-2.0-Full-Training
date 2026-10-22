from motion_mamba2.paths import artifact_path
import os,sys,argparse
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from motion_training import engine,data,batching,streaming,distributed as parallel
from motion_training.io import read as original_read
parallel.setup();batching.install();engine.update=batching.update;data.prepare=streaming.prepare
engine.read=lambda path:streaming.load(path) if Path(path).name=='MANIFEST.json' else original_read(path)
p=argparse.ArgumentParser();p.add_argument('--until',type=int,required=True);p.add_argument('--resume');a=p.parse_args()
engine.train(str(D/'configs/train.json'),str(D/'train'),until=a.until,resume=a.resume,category='B')
