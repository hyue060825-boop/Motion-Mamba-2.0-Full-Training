from common import *
import argparse, types, traceback, contextlib, statistics
import numpy as np
import torch
import adapter
from motion_training import runtime, evaluation, sampling, acceleration
from motion_training.io import identity, sha, copy_state
from g1_motion_contract import decode_committed_window
from g1_bounded_codec import encode_windows
from g1_online_controls import conditions

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
dest = Path(a.output)
dest.mkdir(parents=True, exist_ok=True)
architecture = read(ROOT/'PLAN.json')['architecture']
adapter.install()
original_load = runtime.load
native_prime = sampling.prime_gt
native_payload = sampling.payload


class ExplicitHistoryCounter:
    """Counts actual appended windows, without pretending to own SSM state."""
    def reset(self):
        self._accepted_commits = self._completed_windows = self._next_chunk_id = 0


class ExplicitMOTION_MAMBA2:
    """Original Transformer window path with a 196-frame explicit prefix."""
    def __init__(self, m, c, probe, text):
        self.m, self.c, self.p, self.text = m, c.controller, probe, text
        self.mask = torch.ones(text.shape[:2], device=m.device, dtype=torch.bool)
        self.reset()

    def reset(self):
        self.c.reset()
        self.translation = torch.zeros(1, 3, device=self.m.device)
        self.heading = torch.zeros(1, device=self.m.device)
        self.previous = self.physical = self.history = None
        self.i = 0

    def step(self, noise, control=None, future_observation=None):
        m = self.m
        self.p.begin(noise)
        obs, mask = conditions(m, control, self.translation)
        if future_observation is not None:
            obs, mask = future_observation
        n = 0 if self.history is None else self.history.shape[1]*4
        if n:
            obs = torch.cat((torch.zeros(1, n, 414, device=m.device), obs), 1)
            mask = torch.cat((torch.zeros(1, n, 414, device=m.device, dtype=torch.bool), mask), 1)
        sequence = m._generate_window(
            self.history, self.translation, 0, n, obs.shape[1], self.text, self.mask,
            self.heading, mask, obs, torch.tensor([10], device=m.device),
            (2., 2.), list(range(9, -1, -1)), progress_bar=lambda x: x,
            target_motion=None, cfg_type=None)
        sequence, center, root = m._recenter_history(
            sequence, torch.tensor([sequence.shape[1]*4-1], device=m.device), requantize=True)
        self.translation += center
        # Exactly the original cropped-history first-heading rule.
        headings = m.motion_rep.get_root_heading_angle(m.motion_rep.global_root_stats.unnormalize(root))
        self.heading = headings[:, max(0, root.shape[1]-196)]
        self.history = sequence[:, -49:].clone()
        world = sampling.world_motion(m, sequence[:, -13:], self.translation)
        decoded, self.previous = decode_committed_window(m, world, self.previous)
        physical = m.motion_rep.unnormalize(decoded)
        self.physical = physical if self.physical is None else torch.cat((self.physical, physical), 1)
        self.i += 1
        self.c._accepted_commits += 1
        self.c._completed_windows = self.c._next_chunk_id = self.i
        row, qpos = sampling.measure(m, decoded)
        return row, qpos, decoded, world, sequence[:, -13:]


def explicit_prime(m, c, probe, physical, text):
    runner = ExplicitMOTION_MAMBA2(m, c, probe, text)
    normalized = m.motion_rep.normalize(physical)
    pad = torch.ones(normalized.shape[:2], device=m.device, dtype=torch.bool)
    tokens, _ = encode_windows(m.hybrid, normalized, pad.sum(1), pad)
    runner.history, center, _ = m._recenter_history(
        tokens, torch.tensor([physical.shape[1]-1], device=m.device), requantize=False)
    runner.history = runner.history[:, -49:]
    runner.translation = center
    runner.heading = m.motion_rep.get_root_heading_angle(physical)[:, 0]
    runner.previous = tokens[:, -m.decoder_history_frames//4:].clone()
    runner.physical = physical.clone()
    runner.i = physical.shape[1]//52
    runner.c._accepted_commits = runner.c._completed_windows = runner.c._next_chunk_id = runner.i
    return runner, [dict(frames_end=physical.shape[1], latent=identity(tokens),
                        kind='explicit_history_not_persistent_state')]


cfg = read(LATE/'evaluation/100000/config.json')
cfg.update(parent=a.checkpoint, budget_root=str(dest/'budget'), evaluation_split='DEV',
           evaluation_source_indices=[3, 4], kernel_gate_path=str(dest/'NO_KERNEL_ADMISSION.json'))
write(dest/'budget/LIMITS.json', dict(gpu_hours=2, wall_hours=2))


def load(*args, **kwargs):
    m, c, ck = original_load(*args, **kwargs)
    ck = dict(ck, config=dict(ck['config']))
    ck['config']['joint_limit_spec'] = cfg['common_joint_limit_spec']
    m.diffusion.calc_diffusion_vars(m.diffusion.space_timesteps(10)[0])
    if architecture == 'official_motion_mamba2_transformer':
        c.controller = ExplicitHistoryCounter()
    write(dest/'LOAD.json', dict(checkpoint=str(a.checkpoint), sha256=sha(a.checkpoint),
          architecture=architecture, step=ck['step'], weights_unmodified=True,
          decoder_history_frames=m.decoder_history_frames,
          sequence_history='explicit_196' if architecture=='official_motion_mamba2_transformer' else 'accepted_Minimal_W4_state',
          rank_states_exact=len(set(ck['framework']['rank_state_hashes']))==1))
    return m, c, ck


runtime.load = load
# These graphs target Hybrid; reference inference is used for both baselines.
acceleration.admitted_backend = lambda c, cfg: dict(backend='reference', gate=None,
                                                    reason='Architecture-specific Hybrid kernels not applied')
acceleration.stats = lambda c: dict(reference=True, hybrid_kernel_used=False)
if architecture == 'official_motion_mamba2_transformer':
    sampling.Native = ExplicitMOTION_MAMBA2
    sampling.prime_gt = explicit_prime
    def explicit_payload(runner, chunks, text, control=None):
        value = native_payload(runner, chunks, text, control)
        value['explicit_history'] = copy_state(runner.history)
        return value
    sampling.payload = explicit_payload

write(dest/'PROTOCOL.json', dict(architecture=architecture, scope='DEV96 foundation + DEV16 continuous chains',
      benchmark_config_source=str(LATE/'evaluation/100000/config.json'),
      semantic_and_geometry_thresholds_unchanged=True, same_prompts_sources_noise_and_CFG=True,
      official_MOTION_MAMBA2_explicit_history_frames=196, Minimal_actual_state_commit=True,
      inference='reference', physics=False, actual_ui=False, checkpoint_sha256=sha(a.checkpoint)))
config = dest/'config.json'
write(config, cfg)
try:
    if a.smoke:
        m, c, ck = load(a.checkpoint, False)
        probe = sampling.NoiseProbe(m)
        try:
            with torch.inference_mode():
                result = evaluation.foundation(m, c, ck, probe, cfg, dest/'smoke', True)
            assert result['windows']==48
            write(dest/'SMOKE.json', dict(passed=True, windows=result['windows']))
        finally:
            probe.close()
    else:
        result = evaluation.evaluate(a.checkpoint, str(dest/'DEV'), str(config))
        assert result['complete'] and len(result['foundation']['rows'])==96 and len(result['rows'])==16
        foundation = result['foundation']['rows']
        summarize = lambda rows: dict(n=len(rows), function=sum(x['function_pass'] for x in rows),
                                    joint=sum(x['joint_pass'] for x in rows))
        slip_key = 'metrics.whole.contact_slip_percent'
        summary = dict(checkpoint=str(a.checkpoint), sha256=result['checkpoint_sha256'], architecture=architecture,
            step=read(dest/'LOAD.json')['step'], foundation=summarize(foundation),
            cold_text=summarize([x for x in foundation if x['cell']=='cold_text']),
            actions={name:summarize([x for x in foundation if x['action']==name]) for name in sorted({x['action'] for x in foundation})},
            average_foundation_slip=statistics.mean([x['metrics']['whole']['contact_slip_percent'] for x in foundation]), slip_key=slip_key,
            continuous=dict(n=16,function=sum(x['function_pass'] for x in result['rows']),joint=sum(x['all_pass'] for x in result['rows'])),
            scope='DEV; no physics or actual UI; models at unequal training steps, not a final architecture comparison')
        write(dest/'SUMMARY.json', summary)
        print(json.dumps(summary), flush=True)
except Exception:
    write(dest/'FAILURE.json', dict(error=traceback.format_exc(), time=time.time()))
    raise
