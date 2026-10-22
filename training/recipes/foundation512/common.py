from motion_mamba2.paths import artifact_path
import os, sys, json, time
from pathlib import Path
OLD = Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
ROOT = OLD / 'batch512_after75k_20261007'
PYTHON = artifact_path('python_9f8003379b')
sys.path[:0] = [str(ROOT/'code/src/motion_training/_runtime'), str(ROOT/'code/src'),
                artifact_path('mamba_op_triton_3220f2c818'), artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n'); tmp.replace(p)
def status(stage, **details):
    write(ROOT/'STATUS.json', dict(stage=stage, time=time.time(), controller_pid=os.getpid(), **details))
    print(stage, details, flush=True)
