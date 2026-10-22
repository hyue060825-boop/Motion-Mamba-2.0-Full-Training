from motion_mamba2.paths import artifact_path
from common import *
import traceback,hashlib
import torch
from external_model import ExternalDenoiser
from motion_training.io import sha
try:
 torch.set_num_threads(2);torch.manual_seed(123)
 raw=ExternalDenoiser('motionmamba',SUITE/'repos/MotionMamba',None,None)
 assert raw.scan_counts==[9,7,5,3,1,9,7,5,3,1],raw.scan_counts
 # ModuleList order follows the source's reversed decoder placement; retain it.
 cfg=read(ROOT/'configs/batch32.json');late=read(ROOT/'configs/batch512.json')
 reference=Path(artifact_path('cmdm_g1_7949ac0b80'))
 fields=read(ROOT/'gates/PROTOCOL.json')['matched_fields']
 for b,c in [(32,cfg),(512,late)]:
  old=read(reference/f'configs/batch{b}.json');assert all(c[k]==old[k] for k in fields)
 assert cfg['batch_size']==32 and cfg['per_rank_microbatch']==8 and late['batch_size']==512 and late['per_rank_microbatch']==64
 assert sum(p.numel() for p in raw.parameters())>1000000
 assert all(torch.isfinite(p).all() for p in raw.parameters())
 info=dict(passed=True,actual_Mamba_modules=20,temporal_scan_counts=raw.scan_counts,capacity=256,hidden_width=256,core_layers=11,
  parameters=sum(p.numel() for p in raw.parameters()),active_parameters=sum(p.numel() for p in raw.parameters() if p.requires_grad),inactive_names=raw.inactive_prefixes,
  fresh_init_seed=123,official_or_Native_denoiser_weights_loaded=False,shared_frozen_G1_codec=True,source_adaptations=raw.source_edits,
  schedule=[[0,75000,32],[75000,100000,512]],exposures=15200000,matched_fields=fields,cpu_is_not_training_or_gpu_gate=True,time=time.time())
 write(ROOT/'gates/CPU_PREFLIGHT.json',info)
 sources={str(p.relative_to(SUITE)):sha(p) for p in (SUITE/'scripts').glob('*.py')}
 sources.update({str(p.relative_to(SUITE)):sha(p) for p in (SUITE/'repos/MotionMamba').rglob('*.py') if not p.name.startswith('._')})
 sources.update({str(p.relative_to(SUITE)):sha(p) for p in (ROOT/'configs').glob('*.json')})
 sources.update({str(p.relative_to(SUITE)):sha(p) for p in (SUITE/'code').rglob('*.py')})
 for p in (SUITE/'deps').glob('*.so'):sources[str(p.relative_to(SUITE))]=sha(p)
 write(ROOT/'SOURCE_HASHES.json',sources)
 print(json.dumps(info,ensure_ascii=False),flush=True)
except BaseException:write(ROOT/'gates/CPU_PREFLIGHT_FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
