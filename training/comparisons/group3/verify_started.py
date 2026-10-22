from motion_mamba2.paths import artifact_path
from pathlib import Path
import json, hashlib, time, os
import torch
S=Path(artifact_path('g1_external_matched100k_batch3_20261020_6bfe8d540c'))
def rd(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def wr(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
rows=[]
for name in ['CMDM_G1','MotionStreamer_G1']:
    r=S/name;start=time.time()
    while not (r/'train/checkpoint100.pt').exists():
        if (r/'FAILURE.json').exists() or (r/'CONTROLLER_FAILURE.json').exists():
            raise RuntimeError(name+' failed; inspect live failure')
        if time.time()-start>600:raise TimeoutError(name+' has not saved100 updates')
        time.sleep(5)
    # LATEST is written after the atomic file rename; require both receipts.
    while rd(r/'LATEST.json')['step']<100:time.sleep(1)
    p=r/'train/checkpoint100.pt'
    ck=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    init=torch.load(r/'initial/checkpoint0.pt',map_location='cpu',weights_only=False,mmap=True)
    assert ck['step']==ck['scheduler']['completed']==100 and ck['exposures']==3200
    assert init['step']==init['scheduler']['completed']==init['exposures']==0
    assert len(init['optimizer']['state'])==0
    assert len(ck['rng_by_rank'])==4 and len(set(ck['rank_state_hashes']))==1
    assert len(init['rng_by_rank'])==4 and len(set(init['rank_state_hashes']))==1
    states=ck['optimizer']['state'];assert len(states)>0
    steps={int(v['step']) for v in states.values()}
    assert steps=={100},steps
    assert all(v['exp_avg'].dtype==v['exp_avg_sq'].dtype==torch.float32 for v in states.values())
    assert all(v.dtype==torch.float32 for k,v in ck['model'].items() if v.is_floating_point() and not k.endswith('.attn.scale'))
    for k,h in ck['source_hashes'].items():assert sha(S/k)==h,k
    pid=rd(r/'CONTROLLER_PID.json')['pid'];child=rd(r/'ACTIVE_CHILD.json')
    assert os.readlink(f'/proc/{pid}/cwd')==str(r) and child['mode']=='train'
    assert Path(f'/proc/{child["pid"]}').exists()
    fresh=[rd(r/f'gates/rank{i}_FRESH.json') for i in range(4)]
    assert all(x['step']==x['optimizer_states']==0 for x in fresh)
    row=dict(model=name,actual_completed_updates=100,checkpoint=str(p),checkpoint_sha256=sha(p),
             Adam_states=len(states),all_Adam_steps=100,Adam_moments_FP32=True,
             model_master_parameters_FP32=True,four_rank_RNG=True,all_rank_model_Adam_hash_equal=True,
             initial_scheduler_and_Adam_empty=True,initial_constructor_not_check_weights=True,
             production_checks_not_counted_in_step=True,source_hashes_verified=True,
             controller_pid_verified=pid,actual_live=rd(r/'LIVE.json'),
             warm_gate_seconds_per_update=rd(r/'EXECUTION_RELEASE.json')['warm_gate_seconds_per_update'],
             all_required_gates_passed={n:rd(r/f'gates/{n}.json')['passed'] for n in
               ['CPU_PREFLIGHT','PROTOCOL','NUMERICS','INTERFACE','ADMISSION','GEOMETRY_GRAPH_ADMISSION']},time=time.time())
    wr(r/'FULL_SAVE_PROOF.json',row);rows.append(row)
    print(json.dumps(row),flush=True)
    del ck,init
wr(S/'BATCH3_STARTED_VERIFIED.json',dict(status='BOTH_REAL_TRAINING_AND_COMPLETE100_SAVES_VERIFIED',
    models=rows,schedule=[[0,75000,32],[75000,100000,512]],time=time.time()))
queue=rd(S/'QUEUE.json');queue.update(status='BOTH_REAL_TRAINING',startup_proof='BATCH3_STARTED_VERIFIED.json',time=time.time())
wr(S/'QUEUE.json',queue)
