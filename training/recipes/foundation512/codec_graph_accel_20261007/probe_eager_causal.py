"""Isolated attribution probe: same B1 codec plus explicit causal flag, no graph."""
from motion_mamba2.paths import artifact_path
import sys
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'codec_graph_accel_20261007'
sys.path.insert(0,str(R));from common import *
import torch
from codec_graph import FrozenSerialCodecGraphs
def eager_encode(self,motion):
    assert not torch.is_grad_enabled() and not self.converter.autoencoder.training
    pad=torch.ones(motion.shape[:2],dtype=torch.bool,device=motion.device)
    return self._serial(motion,pad)
FrozenSerialCodecGraphs.encode=eager_encode
exec(compile((S/'probe_pipeline.py').read_text(),str(S/'probe_pipeline.py'),'exec'))
