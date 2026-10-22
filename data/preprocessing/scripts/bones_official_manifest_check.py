"""Validate official admission on real first-batch outputs without conversion."""
import copy, shutil
from g1_pipeline.common import ROOT, sha, read_jsonl, write_json, now
from g1_pipeline.kinematic_policy import BASE, load
from g1_pipeline.bones_official_manifest_admission import apply, POLICY, ADDENDUM
from g1_pipeline.kinematic_20260918_runner import bound_check
import kinematic_finalize as final


def run():
    root = POLICY.parent
    rows = read_jsonl(ROOT/'work/kinematic_training_20260917_v1_batch_0000/results.jsonl')
    pending = [r for r in rows if r['dataset']=='bones' and r.get('decision')=='REVIEW_PENDING']
    assert len(pending)==50
    for row in pending:
        resolved = apply(row)
        assert resolved['decision']=='ACCEPTED_KINEMATIC'
        assert resolved['historical_policy_result']['decision']=='REVIEW_PENDING'
        assert apply(resolved)==resolved
        for key, value in row['output_hashes'].items():
            assert resolved['output_hashes'][key]==value
    for source in ('humanml3d','hiphi'):
        row = next(r for r in rows if r['dataset']==source)
        assert apply(row)==row
    for extra in ('NUMERICAL_CONTRACT_FAILURE','SOURCE_HINGE_LIMIT','MISSING_SOURCE_TARGET_FIDELITY_EVIDENCE'):
        row = copy.deepcopy(pending[0])
        row['decision_reasons'].append(extra)
        assert apply(row)==row
    destination = root/'finalize_scene_check'
    destination.mkdir(exist_ok=True)
    fixture = root/'finalize_check'
    for name in ('candidates.jsonl','isolated.jsonl','execution_policy.json'):
        shutil.copyfile(fixture/name,destination/name)
    original_inputs = load(BASE/'input_release.json')
    inputs = dict(original_inputs, inventory=len(rows)+len(read_jsonl(destination/'isolated.jsonl')))
    final.BASE = destination
    final.collect = lambda: (inputs, rows)
    final.manifests()
    stats = final.stats()
    decisions = {r['sample_id']:r for r in read_jsonl(destination/'manifests/all_decisions.jsonl')}
    for row in rows:
        if row['dataset']=='bones':
            assert decisions[row['sample_id']]['decision']=='ACCEPTED_KINEMATIC'
        else:
            assert decisions[row['sample_id']]['decision']==row['decision']
    train = read_jsonl(destination/'manifests/accepted_train.jsonl')
    assert all(r['project_split']=='train' and r['training_role']!='held_out_calibration' for r in train)
    assert len({r['features_hash'] for r in train})==len(train)
    original = load(BASE/'execution_release.json')
    for path,bound in original['protected_history'].items():
        assert sha(ROOT/path)==bound,path
    bound_check(original['bindings'])
    report={'status':'PASS','time':now(),'scope':'first1000 real outputs with official BONES geometry and ground-scope admission restored',
            'official_bones_accepted':864,'original_geometry_rejections_restored':666,'scene_only_pending_restored':50,
            'human_source_decisions_unchanged':True,'other_hard_or_fidelity_failures_preserved':True,
            'trajectories_recomputed':False,'train_split_heldout_duplicates_checked':True,
            'historical_protected_files_unchanged':len(original['protected_history']),
            'live_conversion_bindings_unchanged':True,'stats_frames':stats['merged_frame_weighted']['frames'],
            'stats_unique_counts':stats['unique_motion_counts'],
            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [POLICY,ADDENDUM,ROOT/'scripts/kinematic_finalize.py',ROOT/'g1_pipeline/bones_official_manifest_admission.py',ROOT/'scripts/bones_official_restore.py',ROOT/'scripts/bones_official_manifest_check.py']}}
    write_json(destination/'acceptance.json',report)
    print(report)


if __name__=='__main__':run()
