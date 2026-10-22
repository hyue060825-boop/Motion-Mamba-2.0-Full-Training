"""One-shot internal release after inspecting all inherited frozen gates."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import json, hashlib, os, math, time
S = Path(artifact_path('g1_external_matched100k_batch3_20261020_6bfe8d540c'))
def rd(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p, x):
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
required=['CPU_PREFLIGHT','PROTOCOL','NUMERICS','INTERFACE','ADMISSION','GEOMETRY_GRAPH_ADMISSION']
waiting={'CMDM_G1','MotionStreamer_G1'}
start=time.time()
while waiting and time.time()-start<1800:
    for name in list(waiting):
        r=S/name
        if (r/'CONTROLLER_FAILURE.json').exists():
            raise RuntimeError(f'{name} failed admission; no release: {rd(r/"CONTROLLER_FAILURE.json")}')
        if not all((r/f'gates/{n}.json').exists() for n in required):continue
        if not all(rd(r/f'gates/{n}.json')['passed'] for n in required):
            raise RuntimeError(f'{name}: a frozen gate failed')
        for n,h in rd(r/'SOURCE_HASHES.json').items():assert sha(S/n)==h,n
        plan=rd(r/'PLAN.json');assert plan['schedule']==[[0,75000,32],[75000,100000,512]]
        assert plan['sample_slots']==15200000 and len(plan['cards'])==4
        pid=rd(r/'CONTROLLER_PID.json')['pid']
        assert os.readlink(f'/proc/{pid}/cwd')==str(r)
        assert 'controller.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
        num=rd(r/'gates/NUMERICS.json')
        assert len(num['ranks'])==4
        for rank in num['ranks']:
            assert [x['batch'] for x in rank]==[32,512]
            assert all(x['passed'] and x['fields_exact'] and x['RNG_exact'] for x in rank)
        inter=rd(r/'gates/INTERFACE.json')
        assert inter['completed_generation_windows']==19 and inter['goal_visible_in_every_window']
        assert inter['unobserved_future_GT_taint_delta']==0 and inter['masked_unobserved_values_no_effect']
        assert inter['root_normalizer_roundtrip'] and inter['all414_coordinate_order_exact']
        adm=rd(r/'gates/ADMISSION.json')
        assert adm['world_size']==4 and adm['global_batches']==[32,512]
        assert len(adm['capacity'])==4 and all(len(x)==12 for x in adm['capacity'])
        speed={}
        for batch in [32,512]:
            values=[next(x for x in rank if x['batch']==batch)['warm_median_seconds'] for rank in adm['timings']]
            assert all(math.isfinite(x) and x>0 for x in values)
            speed[batch]=max(values)
        forecast=(75000*speed[32]+25000*speed[512])/3600
        assert forecast < plan['wall_budget_hours'],('forecast exceeds frozen budget',forecast)
        row=dict(passed=True,source_manifest_sha256=sha(r/'SOURCE_HASHES.json'),
                 reviewed_gates={n:sha(r/f'gates/{n}.json') for n in required},
                 actual_controller_pid=pid,full_rank32_512_update_equivalence=True,
                 original_thresholds_unchanged=True,checks_discarded=True,
                 production_must_start_from_constructor_seed123=True,
                 warm_gate_seconds_per_update=speed,forecast_training_hours=forecast,
                 no_other_user_process_terminated=True,time=time.time())
        wr(r/'EXECUTION_RELEASE.json',row)
        print(json.dumps(dict(model=name,**row)),flush=True)
        waiting.remove(name)
    if waiting:time.sleep(5)
if waiting:raise TimeoutError(f'Gates incomplete; no production release: {waiting}')
