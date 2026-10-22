import json
from types import SimpleNamespace
import torch
from g1_pipeline.common import ROOT,setup_imports
from g1_pipeline.tokenizer_normalizer_bridge import FrozenTokenizerBridge
setup_imports()
from motion_mamba2.representation import MotionMamba2MotionRep
from motion_mamba2.skeleton import G1Skeleton34


class RecordingTokenizer(torch.nn.Module):
    num_frames_per_token=4
    encode_with_quantization=True
    encode_with_normalization=True
    def __init__(self,rep):super().__init__();self.motion_rep=rep
    def tokenize(self,x,mask):self.input=x;return x[...,5:]
    def detokenize(self,x,external_cond=None,motion_pad_mask=None):self.external=external_cond;return {'body':x,'root':external_cond}


def test_bridge_preserves_physical_body_and_local_root_across_real_different_stats():
    contract=json.loads((ROOT/'contracts/motion_mamba2_414.json').read_text());assets=json.loads((ROOT/'audit/training_assets_v1.json').read_text())
    skel=G1Skeleton34();project=MotionMamba2MotionRep(skel,25,stats_path=contract['chosen_normalizer'])
    released=MotionMamba2MotionRep(skel,25,stats_path=assets['models']['g1']['snapshot']+'/stats/motion')
    model=RecordingTokenizer(released);bridge=FrozenTokenizerBridge(model,project)
    x=torch.randn(1,8,414,generator=torch.Generator().manual_seed(3));mask=torch.ones(1,8,dtype=torch.bool)
    token=bridge.tokenize(project.normalize(x),mask)
    torch.testing.assert_close(released.unnormalize(model.input),x,atol=1e-5,rtol=1e-5)
    root=torch.randn(1,8,4);local=project.local_root_stats.normalize(root)
    decoded=bridge.detokenize(token,external_cond=local,motion_pad_mask=mask)
    torch.testing.assert_close(project.body_stats.unnormalize(decoded['body']),x[...,5:],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(released.local_root_stats.unnormalize(model.external),root,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(project.local_root_stats.unnormalize(decoded['root']),root,atol=1e-5,rtol=1e-5)
