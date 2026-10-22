from common import *
import hashlib,py_compile
p=ROOT/'code/src/motion_training/engine.py';source=p.read_text()
old="            saved=ck['framework'];assert saved['recipe_hash']==recipe_hash(cfg),'Recipe changed: explicit new stage required'"
new="""            saved=ck['framework']
            if saved['recipe_hash']!=recipe_hash(cfg):
                gatepath=os.environ.get('MOTION_MICROBATCH_GATE')
                assert gatepath,'Recipe changed: explicit admitted microbatch transition required'
                gate=read(gatepath);before=saved['effective_recipe']
                assert gate['passed'] and gate['kind']=='micro16_to64_same_global512'
                assert str(cp)==gate['resume_checkpoint'] and sha(cp)==gate['resume_sha256']
                assert saved['recipe_hash']==gate['old_recipe_hash'] and recipe_hash(cfg)==gate['new_recipe_hash']
                diff={k for k in set(before)|set(cfg) if before.get(k)!=cfg.get(k)}
                assert diff=={'per_rank_microbatch'} and before['per_rank_microbatch']==16 and cfg['per_rank_microbatch']==64
                assert before['batch_size']==cfg['batch_size']==512
                ck['_resource_transition']=dict(kind=gate['kind'],gate=gatepath,global_batch=512,old_micro=16,new_micro=64,optimizer_and_scheduler_inherited=True)
"""
assert old in source;before=hashlib.sha256(p.read_bytes()).hexdigest();p.write_text(source.replace(old,new));py_compile.compile(str(p),doraise=True)
hashes=read(ROOT/'SOURCE_HASHES.json');hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();write(ROOT/'SOURCE_HASHES.json',hashes)
write(ROOT/'MICROBATCH_RESUME_ADAPTER.json',dict(before=before,after=hashes[str(p)],scope='explicit recipe difference limited to microbatch16to64; global512 and sampler/Adam/schedule unchanged',no_checkpoint_rewriting=True))
print('MICROBATCH_RESUME_ADAPTER_INSTALLED')
