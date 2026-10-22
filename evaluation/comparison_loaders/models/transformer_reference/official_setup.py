from motion_mamba2.paths import artifact_path
from pathlib import Path
import sys,os
TRAIN=Path(artifact_path('g1_motion_mamba2_four_gpu_matched100k_20261009_a4a8fc8513'))
def activate():
 os.environ.update(EVAL_TRAINING_ROOT=str(TRAIN),MOTION_MAMBA2_SEQUENCE_MODEL='transformer',HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 sys.path[:0]=[str(TRAIN/'code/src/motion_training/_runtime'),str(TRAIN/'code/src'),str(TRAIN),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
 import adapter
 return adapter.load
