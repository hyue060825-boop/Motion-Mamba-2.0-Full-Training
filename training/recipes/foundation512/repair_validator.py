import hashlib, py_compile
from common import *
p=ROOT/'code/src/motion_training/engine.py'
text=p.read_text();before=hashlib.sha256(p.read_bytes()).hexdigest()
old="c['batch_size'] in [8,16,32,64,128,256,512]"
new="c['batch_size'] in [8,10,16,32,64,128,256,512]"
assert old in text;p.write_text(text.replace(old,new));py_compile.compile(str(p),doraise=True)
from motion_training import engine
assert engine.config(ROOT/'evaluation/75000/config_DEV.json')['batch_size']==10
admission=read(ROOT/'BATCH_ADMISSION.json');assert admission['passed'] and admission['micro']==16
hashes=read(ROOT/'SOURCE_HASHES.json');hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();write(ROOT/'SOURCE_HASHES.json',hashes)
write(ROOT/'VALIDATOR_REPAIR.json',dict(cause='isolated batch whitelist excluded evaluator batch10',
    before=before,after=hashes[str(p)],evaluation_config_load_passed=True,
    gradient_forward_optimizer_unchanged=True,prior_micro16_admission_reused=True,
    baseline_evaluation_deferred_until_after_continuation=True))
print('EVALUATION_CONFIG_REPAIR_VERIFIED',flush=True)
