from motion_mamba2.paths import artifact_path
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path(artifact_path('batch512_after75k_20261007_c710d5b650'))
PY=artifact_path('python_9f8003379b')
sys.path[:0]=[str(ROOT/'code/src/motion_training/_runtime'),str(ROOT/'code/src'),str(ROOT/'accel'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from motion_training.io import read,write,sha,identity,copy_state,append
def status(stage,**kw):
    write(ROOT/'STATUS.json',dict(stage=stage,time=time.time(),**kw))
    print(stage,kw,flush=True)
