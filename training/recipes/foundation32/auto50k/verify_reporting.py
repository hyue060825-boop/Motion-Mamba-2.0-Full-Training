from motion_mamba2.paths import artifact_path
import hashlib,json
from pathlib import Path
from report50k import foot_summary,evaluate_report
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sample=[dict(frames=52,contact_slip_percent=10.,legacy_skate_percent=5.,contact_fraction=.5,contact_speed_m_s=.1),dict(frames=52,contact_slip_percent=30.,legacy_skate_percent=30.,contact_fraction=1.,contact_speed_m_s=.3),dict(frames=52,contact_slip_percent=None,legacy_skate_percent=0.,contact_fraction=0.,contact_speed_m_s=None)]
r=foot_summary(sample)
assert r['windows']==3 and r['valid_slip_windows']==2 and r['undefined_slip_windows']==1
assert abs(r['mean_contact_slip_percent']-20)<1e-10
assert abs(r['pooled_contact_slip_percent']-100*35/150)<1e-10
assert abs(r['mean_legacy_skate_percent']-35/3)<1e-10
assert abs(r['mean_contact_fraction']-.5)<1e-10
rejected=False
try:evaluate_report(artifact_path('new20k_ff7914f07b'),artifact_path('load_evaluate_json_f47ed93cda'),D/'auto50k/REJECTION_TEST')
except AssertionError as e:
 assert 'stored steps' in str(e);rejected=True
assert rejected,'20k must not pass as50k'
original=json.loads((D/'EXECUTED_SOURCE_HASHES.json').read_text())
changed=[p for p,h in original.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
transition=json.loads((D/'topology4/SOURCE_TRANSITION.json').read_text())
assert changed==[x['path'] for x in transition['changes']],changed
for item in transition['changes']:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['after']
result=dict(aggregation=True,undefined_not_zero=True,contact_weighting=True,reject_wrong_step=True,training_code_and_configs_unchanged=True,original_resource_transition_preserved=True,old_vs_old_self_test=json.loads((D/'auto50k/SELF_TEST/SUMMARY.json').read_text())['delta_new_minus_old'])
(D/'auto50k/VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
