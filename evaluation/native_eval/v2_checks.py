"""Run regression without initializing any server GPU."""
from motion_mamba2.paths import artifact_path
import os
import subprocess
import sys
from .common import *

def run():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    p=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=CODE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    output_path(ROOT/'IMPLEMENTATION_TESTS_V2.log').write_text(p.stdout)
    write(ROOT/'IMPLEMENTATION_TESTS_V2.json',dict(passed=p.returncode==0,exit_code=p.returncode,output=p.stdout,code={str(f.relative_to(CODE)):sha(f) for f in (CODE/'native_eval').glob('*.py')},tests={str(f.relative_to(CODE)):sha(f) for f in (CODE/'tests').glob('*.py')},gpu_visible='none',time=time.time()))
    prior=read(artifact_path('evaluator_asset_manifest_json_c38780c4a2'));actual={name:sha(TEXT/name) for name in ['config.json','model.safetensors','spiece.model']}
    if actual!=prior['text_hashes']:raise ValueError('Actual T5 assets changed')
    write(ROOT/'TEXT_BACKBONE_ACTUAL_HASHES.json',dict(actual=actual,revision=prior['text_revision'],passed=True))
    if p.returncode:raise RuntimeError(p.stdout)
    return p.stdout

if __name__=='__main__':print(run())
