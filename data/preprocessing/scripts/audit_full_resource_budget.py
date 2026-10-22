"""Metadata-based full-batch budget, with measured smoke storage rates."""
import json
import shutil
from collections import Counter
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha


def run():
    manifest=ROOT/'manifests/full_candidates.v2.jsonl';totals={};rows_by_source=Counter();counts=Counter()
    # Only candidate records get output estimates. Unsupported/missing/leakage
    # exclusions keep their original denominator in the inventory.
    with manifest.open() as stream:
        for line in stream:
            row=json.loads(line);source=row['dataset'];rows_by_source[source]+=1;counts[(source,row['eligibility'])]+=1
            group=totals.setdefault(source,{'candidate_clips':0,'known_seconds':0.,'unknown_duration_clips':0})
            if row['eligibility']!='candidate':continue
            group['candidate_clips']+=1
            duration=row['native_end_sec']-row['native_start_sec'] if row.get('native_end_sec') is not None else row.get('metadata_duration_sec')
            if duration is None:group['unknown_duration_clips']+=1
            else:group['known_seconds']+=duration
    sample=ROOT/'work/production_contract_validation_v4/pels_air_punch_001__A493';c=json.loads((sample/'contract.json').read_text());n=c['frames']
    bytes_per_frame={name:(sample/name).stat().st_size/n for name in ('motion.csv','qpos.npy','features.npy')}
    # Original GMR trajectory, post-collision qpos, targets and diagnostic arrays
    # cost extra; verbose per-frame JSON is unsuitable for all production clips.
    raw_overhead=2*36*8+14*3*4+14*2*4
    for source,group in totals.items():
        frames=group['known_seconds']*25;group['estimated_25hz_frames_upper']=frames
        group['main_output_bytes_estimate']=frames*sum(bytes_per_frame.values())
        group['native_diagnostic_bytes_estimate']=frames*raw_overhead if source!='bones' else 0
        group['known_candidate_hours']=group['known_seconds']/3600
    main=sum(g['main_output_bytes_estimate']+g['native_diagnostic_bytes_estimate'] for g in totals.values())
    reserve=50*10**9;free=shutil.disk_usage(ROOT).free
    report={'status':'PLANNING_ESTIMATE_NOT_FULL_BATCH_AUTHORIZATION','manifest_sha256':sha(manifest),'sources':totals,
        'full_inventory_counts':dict(rows_by_source),'eligibility_counts':{s+':'+e:n for (s,e),n in counts.items()},
        'measured_example_bytes_per_frame':bytes_per_frame,'example_frames':n,'sample_contract_sha256':sha(sample/'contract.json'),
        'estimated_main_and_native_diagnostics_gb':main/1e9,'estimated_with_25percent_io_overhead_gb':main*1.25/1e9,
        'free_gb_at_audit':free/1e9,'minimum_shared_disk_reserve_gb':reserve/1e9,
        'all_metadata_candidates_fit_upper_estimate':main*1.25+reserve<free,
        'retarget_cpu_hours_at_bones_measured_0_07sec_per_frame':sum(g['estimated_25hz_frames_upper'] for s,g in totals.items() if s!='bones')*.07/3600,
        'resource_policy':{'initial_workers':1,'max_cpu_workers_after_equivalence_test':8,'gpu_workers':0,
            'native_archive_extraction':'stream requested Body-only members; bound cache; avoid materializing all archives',
            'first_batch_clips':1000,'stop_source_batch_failure_fraction_above':.05,
            'storage_strategy':'NPY float32 features, NPY qpos, strict CSV; compressed diagnostics; complete raw motion retained for rejected/corrected outputs',
            'no_raw_source_deletion':True},
        'limitations':['metadata candidates include scenes that the frozen production scope may exclude',
            'BONES cost rate does not establish source-specific HML/HiPHI throughput; measure after source smoke',
            'source archives, staged copies, text cache and full dynamic trajectories need separate bounded budgets'],
        'required_prior_gate':'04_fusion_qa PASS with matching bindings; no full processing run'}
    write_json(ROOT/'audit/full_resource_budget_v1.json',report)
    print({k:report[k] for k in ['estimated_main_and_native_diagnostics_gb','estimated_with_25percent_io_overhead_gb','free_gb_at_audit','all_metadata_candidates_fit_upper_estimate','retarget_cpu_hours_at_bones_measured_0_07sec_per_frame']})


if __name__=='__main__':run()
