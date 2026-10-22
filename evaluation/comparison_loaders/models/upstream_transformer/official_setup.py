"""Load the untouched official ARDY package before shared data utilities."""
from motion_mamba2.paths import artifact_path
import sys,os
from pathlib import Path
OFFICIAL=Path(artifact_path('motion_mamba2_official_backup_6b3706c1ce'))
UTILITY=Path(artifact_path('g1_motion_mamba2_four_gpu_matched100k_20261009_a4a8fc8513'))
def activate():
    sys.path.insert(0,str(OFFICIAL))
    import motion_mamba2
    assert Path(ardy.__file__).resolve().is_relative_to(OFFICIAL)
    from motion_mamba2.models.load_model import load_model
    sys.path[:0]=[str(UTILITY/'code/src'),str(UTILITY/'code/src/motion_training/_runtime'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
    os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return load_model
