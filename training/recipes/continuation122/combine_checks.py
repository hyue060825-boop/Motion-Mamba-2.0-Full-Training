from motion_mamba2.paths import artifact_path
from pathlib import Path
import py_compile
r=Path(artifact_path('g1_native100k_original117k_protocol_20261014_dbcf4a1353'))
p=r/'profile_extreme.py';s=p.read_text()
marker='    else:\n        for repeat in range(a.rounds):'
assert marker in s
addition='''        # Reuse warmed graphs after strict equivalence, preserving all original state.
        if a.tag=="g0_full512_gate":
            assert all(x["passed"] for x in checks)
            for repeat in range(3):
                for mode in (modes if repeat%2==0 else list(reversed(modes))):
                    restore();row,_=update(mode);discarded+=1;row["repeat"]=repeat
                    rows.extend(parallel.gather(row))
                    if parallel.leader():print("PAIRED",repeat,mode,row["seconds"],flush=True)
            if parallel.leader():
                timing={mode:[max(x["seconds"] for x in rows if x.get("repeat")==i and x["mode"]==mode) for i in range(3)] for mode in modes}
                write(ROOT/"acceleration_extreme/g0_timing/TIMING.json",dict(rows=rows,timing=timing,medians={k:statistics.median(v) for k,v in timing.items()},checkpoint=a.checkpoint,pool=a.pool,step=a.step,all_512_samples_and_all_losses=True,reused_warmup_after_successful_gate=True,discarded_updates=discarded))
'''
s=s.replace(marker,addition+marker);p.write_text(s)
py_compile.compile(str(p),doraise=True)
print('Combined strict gate and matched timing ready')
