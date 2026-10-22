from common import *
import argparse,copy
from motion_training import runtime,evaluation
from g1_padding_contract import install_padding_contract
p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--final',action='store_true');a=p.parse_args()
folder=ROOT/'evaluation'/str(a.step);folder.mkdir(parents=True,exist_ok=True)
cfg=read(OLD/'evaluation/100000/config.json');cfg.update(parent=a.checkpoint,budget_root=str(ROOT/'budget'),kernel_gate_path=str(folder/'NO_ADMISSION.json'))
original_load=runtime.load
def common_load(*args,**kw):
    m,c,ck=original_load(*args,**kw);ck=dict(ck,config=dict(ck['config']))
    ck['config']['joint_limit_spec']=cfg['common_joint_limit_spec']
    install_padding_contract(m,'g1_hybrid_attention_mask_v1');m.generation_contract='g1_committed_v3'
    ck['config'].update(padding_contract='g1_hybrid_attention_mask_v1',generation_contract='g1_committed_v3')
    assert ck['step']==112000+a.step
    write(folder/'LOAD.json',dict(checkpoint=a.checkpoint,sha256=sha(a.checkpoint),global_step=ck['step'],common_metadata_only=True,weights_not_modified=True))
    return m,c,ck
runtime.load=common_load
if a.step==10000:
    from motion_training import acceleration
    import traceback
    gate_dir=folder/'kernel'
    cfg['kernel_gate_path']=str(gate_dir/'GATE.json')
    gate_cfg=folder/'KERNEL_config.json';write(gate_cfg,cfg)
    try:
        gate=acceleration.gate(str(gate_cfg),str(gate_dir))
        assert gate.get('passed') and gate['checkpoint_sha256']==sha(a.checkpoint)
    except Exception:
        write(folder/'KERNEL_FALLBACK.json',dict(reason=traceback.format_exc(),backend='reference',quality_benchmark_unchanged=True))
        cfg['kernel_gate_path']=str(folder/'NO_ADMISSION.json')
        print('5K_KERNEL_NOT_ADMITTED: reference fallback',flush=True)
for split in (['DEV','FINAL'] if a.step==10000 or a.final else ['DEV']):
    c=copy.deepcopy(cfg);c.update(evaluation_split=split,evaluation_source_indices=[10,11] if split=='FINAL' else [3,4])
    path=folder/(split+'_config.json');write(path,c)
    r=evaluation.evaluate(a.checkpoint,str(folder/split),str(path))
    print('EVALUATED',a.step,split,r['foundation']['function'],r['foundation']['joint'],sum(x['all_pass'] for x in r['rows']),flush=True)
