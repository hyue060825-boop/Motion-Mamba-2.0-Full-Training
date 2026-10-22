from motion_mamba2.paths import artifact_path
import ast,sys
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'codec_graph_accel_20261007'
sys.path.insert(0,str(R))
tree=ast.parse((S/'run_probes.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Assign))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<isolated_probe_helpers>','exec'),globals())
launch('eager_causal_4r','probe_eager_causal.py',['--mode','serial_codec','--workers','0','--label','eager_causal_4r'],4)
a=read(S/'probes/reference_4r/RESULT.json');b=read(S/'probes/eager_causal_4r/RESULT.json')
loss_difference=max(abs(x['loss']-y['loss'])/max(abs(x['loss']),1e-8) for x,y in zip(a['rows'],b['rows']))
write(S/'ATTRIBUTION.json',dict(reference_seconds=a['mean_seconds'],eager_causal_seconds=b['mean_seconds'],
    graph_seconds=read(S/'BENCHMARK.json')['confirm_candidate']['mean_seconds'],
    explicit_causal_speedup=a['mean_seconds']/b['mean_seconds'],maximum_loss_relative_difference=loss_difference,
    same_samples=all(x['sampler_identity']==y['sampler_identity'] for x,y in zip(a['rows'],b['rows'])),
    interpretation='Explicit causal flag and host-known uniform history checks together; graph adds submission reduction',
    probe_updates_discarded=True))
assert loss_difference<.001
print('ATTRIBUTION',read(S/'ATTRIBUTION.json'),flush=True)
