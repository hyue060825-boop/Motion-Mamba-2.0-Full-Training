"""Freeze tracking policy from disclosed controller development evidence only."""
import json
from g1_pipeline.common import ROOT,write_json,sha,now
from g1_pipeline.calibration_protocol import require_frozen


def run():
    require_frozen();path=ROOT/'configs/dynamic_protocol.v1.json'
    if path.exists():
        protocol=json.loads(path.read_text())
        for name,bound in protocol['bindings'].items():
            if sha(ROOT/name)!=bound:raise ValueError('FROZEN_DYNAMIC_DEPENDENCY_CHANGED:'+name)
        return protocol
    evidence=ROOT/'audit/sonic_bones_dev3_v1.json';dev=json.loads(evidence.read_text())
    if len(dev['results'])!=9 or not all(r['full_duration_completed'] for r in dev['results']):raise ValueError('CONTROLLER_DEVELOPMENT_INCOMPLETE')
    # Fixed conservative engineering limits, evaluated against the available
    # development evidence before any three-source dynamic panel is run.
    limits={'joint_rmse_rad':.4,'root_translation_rmse_m':.25,'root_orientation_mean_deg':20.,'feet_position_mean_m':.25,
        'reference_relative_height_loss_max_m':.35,'reference_relative_orientation_divergence_deg':60.,'orientation_divergence_duration_s':.2}
    for row in dev['results']:
        if any(row[k]>limits[k] for k in ('joint_rmse_rad','root_translation_rmse_m','root_orientation_mean_deg')) or max(row['feet_position_mean_m'])>limits['feet_position_mean_m']:
            raise ValueError('PROPOSED_ENGINEERING_LIMIT_FAILED_ON_CONTROLLER_DEVELOPMENT')
    paths=[evidence,ROOT/'audit/sonic_controller_static_interface.json',ROOT/'g1_pipeline/dynamic_tracking.py',
        ROOT/'g1_pipeline/sonic_observations.py',ROOT/'g1_pipeline/sonic_scene.py',ROOT/'g1_pipeline/resample.py',
        ROOT/'assets/project_g1_v1/g1.xml',ROOT/'assets/project_g1_sonic_scene_v1/g1.xml']
    interface=json.loads((ROOT/'audit/sonic_controller_static_interface.json').read_text())
    bindings={str(p.relative_to(ROOT)):sha(p) for p in paths};bindings.update(interface['source_hashes'])
    bindings.update({r['path']:r['sha256'] for r in interface['models']})
    protocol={'status':'FROZEN_BEFORE_DYNAMIC_PANEL','time':now(),'bindings':bindings,'per_clip_limits':limits,
        'seeds':[0,1,2],'initial_joint_noise_std_rad':.002,'control_dt_s':.02,'reference_fps':25,
        'supported_scenes':['flat_ground_core_candidate','ground_multisupport'],
        'fall_definition':'height loss versus contemporaneous reference >0.35m, or reference-relative root orientation >60deg for >=0.2s; intentional low postures are not falls',
        'geometry_contact_policy':'all original compiled collision masks/excludes; no new selfcollision exemptions',
        'source_panel_independent_clips_min':30,'source_success_fraction_min':.90,
        'class_matched_success_decline_from_bones_max':.05,'class_matched_joint_rmse_multiplier':1.25,'class_matched_joint_rmse_additive_rad':.02,
        'baseline_basis':'9 development tracks =3 metadata-selected official BONES dev clips x3seeds; all completed, not a 30clip source acceptance panel',
        'threshold_basis':'project engineering bounds frozen before source dynamic evaluation; absolute joint/root/feet fit bounds plus stronger cross-source comparison mandated by user task',
        'uncertainty':'Only 3 independent controller development clips; limited action coverage. This freeze grants no dynamic or hardware PASS.',
        'matched_target_effort_caps':'locked model motor effort ranges; 88Nm hip pitch retained despite SONIC training139Nm discrepancy',
        'hardware_velocity_envelope':None,'hardware_velocity_reason':'No verified matched motor envelope; measure rates and tracking without claiming deployment compliance',
        'full_batch_dynamic_sample':'per source min(N,max(100,ceil(.01*N))) independent accepted families, plus new anomaly/high-risk strata',
        'retune_rule':'preserve failed independent dynamic evaluation, move used clips into development and freeze a new protocol before a fresh disjoint panel'}
    write_json(path,protocol);return protocol


if __name__=='__main__':print(run()['status'])
