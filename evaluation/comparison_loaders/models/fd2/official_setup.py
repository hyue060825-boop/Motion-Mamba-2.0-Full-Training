from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys
SUITE=Path(artifact_path('g1_external_matched100k_batch2_20261019_068631a7db'))
def activate():
 os.environ.update(MOTION_MAMBA2_SEQUENCE_MODEL="transformer",HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
 sys.path[:0]=[str(SUITE/"code/src/motion_training/_runtime"),str(SUITE/"code/src"),str(SUITE/"scripts"),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
