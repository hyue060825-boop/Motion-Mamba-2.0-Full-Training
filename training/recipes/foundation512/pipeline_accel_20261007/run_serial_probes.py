from motion_mamba2.paths import artifact_path
import ast,sys,time
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'pipeline_accel_20261007'
sys.path.insert(0,str(R))
deadline=time.time()+1800
while not (S/'BENCHMARK.json').exists():
    if (S/'ERROR.json').exists():raise RuntimeError('Initial probe failed')
    if time.time()>deadline:raise TimeoutError('Initial benchmark did not finish')
    time.sleep(5)
tree=ast.parse((S/'run_probes.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) or isinstance(n,ast.Assign)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<reuse_probe_functions>','exec'),globals())
launch('serial_input_gate','serial_gate.py')
assert read(S/'SERIAL_INPUT_GATE.json')['passed'],'Serial codec input gate failed; keep already tested loss_only result'
launch('serial_candidate_4r','probe_pipeline.py',['--mode','serial_codec','--workers','4','--label','serial_candidate_4r','--save-gate'],4)
ref=S/'probes/reference_4r';cand=S/'probes/serial_candidate_4r'
g=compare_tensor_dict(ref/'gradients.pt',cand/'gradients.pt');u=compare_tensor_dict(ref/'master_delta.pt',cand/'master_delta.pt')
mom=compare_tensor_dict(ref/'moments.pt',cand/'moments.pt')
a=read(ref/'RESULT.json');b=read(cand/'RESULT.json');ra=read(ref/'RESTORED.json');rb=read(cand/'RESTORED.json')
eq=dict(gradient=g,actual_adam_master_delta=u,adam_moments=mom,starting_states_equal=ra==rb,
    same_samples=all(x['sampler_identity']==y['sampler_identity'] for x,y in zip(a['rows'],b['rows'])),
    loss_relative_difference=abs(a['rows'][0]['loss']-b['rows'][0]['loss'])/max(abs(a['rows'][0]['loss']),1e-8))
eq['passed']=eq['starting_states_equal'] and eq['same_samples'] and eq['loss_relative_difference']<.001 and all(v['cosine']>.999 and v['relative_l2']<.05 for v in [g['total'],u['total'],*u['groups'].values()])
write(S/'SERIAL_UPDATE_GATE.json',eq);assert eq['passed']
candidates=[b]
for workers in [0,2,8]:
    label=f'serial_workers{workers}_4r'
    launch(label,'probe_pipeline.py',['--mode','serial_codec','--workers',str(workers),'--label',label,'--count','12'],4)
    candidates.append(read(S/'probes'/label/'RESULT.json'))
best=min(candidates,key=lambda x:x['mean_seconds'])
launch('serial_confirm_candidate_4r','probe_pipeline.py',['--mode','serial_codec','--workers',str(best['workers']),'--label','serial_confirm_candidate_4r'],4)
launch('serial_confirm_reference_4r','probe_pipeline.py',['--mode','reference','--label','serial_confirm_reference_4r'],4)
ca=read(S/'probes/serial_confirm_candidate_4r/RESULT.json');ra=read(S/'probes/serial_confirm_reference_4r/RESULT.json')
result=dict(reference=a,candidates=candidates,selected=dict(mode='serial_codec',workers=best['workers']),
    confirm_reference=ra,confirm_candidate=ca,mean_speedup=ra['mean_seconds']/ca['mean_seconds'],
    time_reduction_fraction=1-ca['mean_seconds']/ra['mean_seconds'],sample_seconds=ca['mean_seconds'],all_probe_updates_discarded=True,gate_passed=True)
write(S/'SERIAL_BENCHMARK.json',result)
old=read(S/'BENCHMARK.json')
if ca['mean_seconds']<old['sample_seconds']:
    write(S/'LOSS_ONLY_BENCHMARK.json',old);write(S/'LOSS_ONLY_UPDATE_GATE.json',read(S/'UPDATE_GATE.json'))
    write(S/'BENCHMARK.json',result);write(S/'UPDATE_GATE.json',eq)
print('SERIAL_COMPLETE',result['mean_speedup'],flush=True)
