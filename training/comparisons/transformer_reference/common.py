from motion_mamba2.paths import artifact_path
import os,sys,json,time
from pathlib import Path
ROOT=Path(artifact_path('g1_motion_mamba2_four_gpu_matched100k_20261009_a4a8fc8513'))
OLD=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
LATE=OLD/'batch512_after75k_20261007'
PY=artifact_path('python_9f8003379b')
sys.path[:0]=[str(ROOT/'code/src/motion_training/_runtime'),str(ROOT/'code/src'),str(ROOT),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MOTION_MAMBA2_SEQUENCE_MODEL='transformer')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str)+'\n');t.replace(p)
