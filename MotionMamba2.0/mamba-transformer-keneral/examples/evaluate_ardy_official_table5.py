"""Evaluate the ARDY paper Table 5 generation protocol on trained checkpoints.

The paper uses a nine-second sequence with one second of ground-truth initial
history. In-horizon evaluation places one joint-position goal at the end of
every autoregressive window; out-of-horizon evaluation places one goal at the
end of the complete sequence. Learned HumanML3D metrics remain unavailable in
this checkout, so this script reports paired geometry and physical proxies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))
for path in (
    os.environ.get("MAMBA_REPO", "/data3/zts/packages/state-spaces-mamba"),
    os.environ.get("MAMBA_TRITON_OVERLAY", "/data3/zts/packages/mamba_op_triton"),
):
    sys.path.insert(0, path)

from ardy_data import HumanML3DArdyDataset  # noqa: E402
from evaluate_humanml3d_table45 import (  # noqa: E402
    _angle_mae_degrees,
    _rigid_pa_mpjpe,
    _skate_counts,
)
from train_humanml3d_ardy_tiny import _replace_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "minimal", "hybrid"], required=True)
    parser.add_argument("--model", choices=["core", "g1"], required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", default=None)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--protocol", choices=["in_horizon", "out_of_horizon"], required=True)
    parser.add_argument("--total-seconds", type=float, default=9.0)
    parser.add_argument(
        "--selection-min-seconds",
        type=float,
        default=None,
        help="Select the fixed panel from clips at least this long; defaults to total-seconds.",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cfg-weight-text", type=float, default=2.0)
    parser.add_argument("--cfg-weight-constraint", type=float, default=2.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifacts-output", default=None)
    parser.add_argument(
        "--periodic-state-reset-windows", type=int, default=0,
        help="Reset only persistent KV/SSM state every K generated windows; 0 disables.",
    )
    parser.add_argument("--state-diagnostics-output", default=None)
    parser.add_argument("--mamba-output-decay-start-windows", type=int, default=0)
    parser.add_argument("--mamba-output-decay-half-life-windows", type=float, default=0.0)
    parser.add_argument("--mamba-output-decay-min-scale", type=float, default=0.0)
    return parser.parse_args()


def _select_indices(dataset: HumanML3DArdyDataset, frames: int, count: int, seed: int) -> list[int]:
    eligible = [
        index
        for index, record in enumerate(dataset.records)
        if int(record.get("frames", 0)) >= frames
    ]
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} records have at least {frames} frames")
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.choice(eligible, size=count, replace=False)]


def _goal_joint_indices(skeleton) -> list[int]:
    names = [
        skeleton.bone_order_names[skeleton.root_idx],
        skeleton.left_hand_joint_names[0],
        skeleton.right_hand_joint_names[0],
        skeleton.left_foot_joint_names[0],
        skeleton.right_foot_joint_names[0],
    ]
    return [int(skeleton.bone_index[name]) for name in names]


def _stable_goal_joint(sample_id: str, goal_number: int, candidates: list[int], seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:goal:{goal_number}".encode()).digest()
    return candidates[int.from_bytes(digest[:8], "little") % len(candidates)]


def _build_official_sample(
    dataset: HumanML3DArdyDataset,
    index: int,
    augmentation_seed: int,
    *,
    total_frames: int,
    selection_span_frames: int,
    history_frames: int,
    horizon: int,
    protocol: str,
    motion_rep,
    skeleton,
) -> dict:
    record = dataset.records[index]
    features = torch.from_numpy(np.load(record["feature_path"])).float()
    if len(features) < total_frames:
        raise ValueError(f"sample {record['id']} is shorter than {total_frames} frames")
    if len(features) < selection_span_frames:
        raise ValueError(
            f"sample {record['id']} is shorter than selection span {selection_span_frames}"
        )
    rng = np.random.default_rng(augmentation_seed)
    start = int(rng.integers(len(features) - selection_span_frames + 1))
    physical = features[start:start + total_frames].clone()

    # Match ARDY's initial-history convention: the final prefix pose is the
    # planar origin for both the history prime and the generated target tail.
    center = physical[history_frames - 1, [0, 2]].clone()
    physical[:, 0] -= center[0]
    physical[:, 2] -= center[1]
    normalized = (physical - dataset.mean) / torch.sqrt(dataset.std.square() + 1e-5)

    generation_frames = total_frames - history_frames
    if protocol == "in_horizon":
        goal_frames = list(range(horizon - 1, generation_frames, horizon))
        if not goal_frames or goal_frames[-1] != generation_frames - 1:
            goal_frames.append(generation_frames - 1)
    else:
        goal_frames = [generation_frames - 1]
    candidates = _goal_joint_indices(skeleton)
    goal_joints = [
        _stable_goal_joint(record["id"], number, candidates, augmentation_seed)
        for number in range(len(goal_frames))
    ]

    observed = torch.zeros_like(normalized)
    motion_mask = torch.zeros_like(normalized, dtype=torch.bool)
    root_slice = motion_rep.slice_dict["root_pos"]
    heading_slice = motion_rep.slice_dict["global_root_heading"]
    local_slice = motion_rep.slice_dict["local_joints_positions"]
    for relative_frame, joint in zip(goal_frames, goal_joints):
        frame = history_frames + relative_frame
        if joint == skeleton.root_idx:
            # Pelvis goals follow the root-waypoint training contract. Root
            # height is not exposed as an independent user control.
            motion_mask[frame, root_slice.start] = True
            motion_mask[frame, root_slice.start + 2] = True
        else:
            motion_mask[frame, root_slice] = True
            motion_mask[frame, heading_slice] = True
            offset = local_slice.start + (joint - 1) * 3
            motion_mask[frame, offset:offset + 3] = True
    observed[motion_mask] = normalized[motion_mask]

    text = (
        dataset.text_cache_caption[record["id"]]
        if dataset.text_cache_caption is not None
        else dataset._caption(record, rng)
    )
    output = {
        "id": record["id"],
        "motion": normalized,
        "observed_motion": observed,
        "motion_mask": motion_mask.float(),
        "text": text,
        "goal_frames": torch.tensor(goal_frames, dtype=torch.long),
        "goal_joints": torch.tensor(goal_joints, dtype=torch.long),
    }
    if dataset.text_cache is not None:
        row = dataset.text_cache_index[record["id"]]
        output["text_feature"] = torch.from_numpy(
            np.array(dataset.text_cache[row], copy=True)
        ).float()
    return output


def _load_checkpoint(model, mode: str, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != mode:
        raise ValueError(f"checkpoint mode={checkpoint.get('mode')!r}, requested mode={mode!r}")
    raw = model.denoiser.model
    raw.root_model.load_state_dict(checkpoint["root_model"], strict=True)
    raw.body_model.load_state_dict(checkpoint["body_model"], strict=True)
    auxiliary = checkpoint.get("denoiser_aux")
    if auxiliary is not None:
        current = raw.state_dict()
        unknown = set(auxiliary) - set(current)
        if unknown:
            raise ValueError(f"checkpoint has unknown denoiser_aux keys: {sorted(unknown)}")
        with torch.no_grad():
            for key, value in auxiliary.items():
                current[key].copy_(value.to(device=current[key].device, dtype=current[key].dtype))
    return checkpoint


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch size must be positive")
    if args.total_seconds <= 1.0:
        raise ValueError("total-seconds must exceed the one-second history prefix")
    if args.periodic_state_reset_windows < 0:
        raise ValueError("periodic-state-reset-windows must be non-negative")
    if (
        args.mamba_output_decay_start_windows < 0
        or args.mamba_output_decay_half_life_windows < 0
    ):
        raise ValueError("Mamba output decay settings must be non-negative")
    if not 0.0 <= args.mamba_output_decay_min_scale <= 1.0:
        raise ValueError("Mamba output decay minimum scale must be in [0, 1]")
    if args.mode == "original" and (
        args.periodic_state_reset_windows
        or args.mamba_output_decay_half_life_windows
    ):
        raise ValueError("native ARDY has no adapter-owned recurrent state")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    from ardy.model import load_model
    from ardy.motion_rep.tools import length_to_mask
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model(args.model, device=str(device), text_encoder=False)
    _replace_stats(model, args.stats_dir)
    if args.mode == "original":
        client = model
    else:
        client = install_ardy_sequence_kernel(
            model,
            ArdySequenceKernelConfig(
                mode=args.mode,
                attention_backend="sdpa",
                window_size_chunks=4,
                num_sink_tokens=4,
            ),
        )
        client.controller.configure_rollout_diagnostics(
            enabled=bool(args.state_diagnostics_output),
            periodic_reset_windows=args.periodic_state_reset_windows,
        )
        client.controller.configure_mamba_output_decay(
            start_windows=args.mamba_output_decay_start_windows,
            half_life_windows=args.mamba_output_decay_half_life_windows,
            min_scale=args.mamba_output_decay_min_scale,
        )
    checkpoint = _load_checkpoint(model, args.mode, args.checkpoint, device)
    if checkpoint.get("model", args.model) != args.model:
        raise ValueError(f"checkpoint model={checkpoint.get('model')!r}, requested model={args.model!r}")
    raw = model.denoiser.model
    raw.root_model.eval()
    raw.body_model.eval()

    fps = int(model.motion_rep.fps)
    horizon = int(model.gen_horizon_len)
    total_frames = int(round(args.total_seconds * fps))
    nfpt = int(model.num_frames_per_token)
    history_frames = int(round(1.0 * fps / nfpt)) * nfpt
    generation_frames = total_frames - history_frames
    dataset = HumanML3DArdyDataset(
        args.manifest,
        args.stats_dir,
        text_root=args.text_root,
        frames=total_frames,
        horizon=horizon,
        condition_dropout=0.0,
        seed=args.seed,
        text_cache_dir=args.text_cache,
        history_probability=0.0,
    )
    selection_min_seconds = (
        args.total_seconds
        if args.selection_min_seconds is None
        else args.selection_min_seconds
    )
    selection_min_frames = int(round(selection_min_seconds * fps))
    if selection_min_frames < total_frames:
        raise ValueError("selection-min-seconds must be at least total-seconds")
    indices = _select_indices(dataset, selection_min_frames, args.samples, args.seed)
    samples = [
        _build_official_sample(
            dataset,
            index,
            args.seed + 1000003 * offset,
            total_frames=total_frames,
            selection_span_frames=selection_min_frames,
            history_frames=history_frames,
            horizon=horizon,
            protocol=args.protocol,
            motion_rep=model.motion_rep,
            skeleton=model.skeleton,
        )
        for offset, index in enumerate(indices)
    ]

    constraint_errors = []
    root_speeds = []
    boundary_root_speeds = []
    latencies = []
    sliding = contacts = 0
    root_ade_cm = []
    root_fde_cm = []
    pa_mpjpe_cm = []
    jitter_m_per_frame2 = []
    angle_mae_deg = []
    max_root_speed_event = {"speed": float("-inf"), "sample_id": None, "frame": None}
    artifact_generated_joints = []
    artifact_target_joints = []
    artifact_generated_root = []
    artifact_target_root = []
    artifact_history_joints = []
    artifact_ids = []
    artifact_texts = []
    artifact_goal_frames = []
    artifact_goal_joints = []
    state_diagnostics = []
    num_windows = math.ceil(generation_frames / horizon)
    window_root_errors = [[] for _ in range(num_windows)]
    window_joint_errors = [[] for _ in range(num_windows)]
    window_constraint_errors = [[] for _ in range(num_windows)]
    foot_indices = list(model.skeleton.foot_joint_idx)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    for start in range(0, len(samples), args.batch_size):
        items = samples[start:start + args.batch_size]
        batch = default_collate(items)
        size = len(items)
        full_motion = batch["motion"].to(device)
        full_mask = batch["motion_mask"].to(device)
        full_observed = batch["observed_motion"].to(device)
        text_feat = batch["text_feature"].to(device)[:, None]
        text_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=device)
        init_history = full_motion[:, :history_frames]
        tail_motion = full_motion[:, history_frames:]
        tail_mask = full_mask[:, history_frames:]
        tail_observed = full_observed[:, history_frames:]
        physical_history = model.motion_rep.unnormalize(init_history)
        next_heading = torch.atan2(
            physical_history[:, -1, 4], physical_history[:, -1, 3]
        )

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            if args.mode == "original":
                lengths = torch.full((size,), total_frames, dtype=torch.long, device=device)
                prediction = model(
                    list(batch["text"]),
                    total_frames,
                    num_denoising_steps=args.diffusion_steps,
                    pad_mask=length_to_mask(lengths),
                    first_heading_angle=None,
                    motion_mask=full_mask,
                    observed_motion=full_observed,
                    cfg_weight=(args.cfg_weight_text, args.cfg_weight_constraint),
                    crop_history_length=None,
                    init_history_sequence=init_history,
                    text_feat=text_feat,
                    text_pad_mask=text_mask,
                    progress_bar=lambda values: values,
                )[:, history_frames:]
            else:
                client.reset_memory()
                history_lengths = torch.full(
                    (size,), history_frames, dtype=torch.long, device=device
                )
                history_pad = torch.ones(
                    size, history_frames, dtype=torch.bool, device=device
                )
                history_hybrid, _ = model.hybrid.get_hybrid_motion_from_explicit(
                    init_history, history_lengths, history_pad
                )
                client.controller.prime_accepted_history(
                    history_hybrid,
                    text_feat=text_feat,
                    text_pad_mask=text_mask,
                    first_heading_angle=next_heading,
                )
                lengths = torch.full(
                    (size,), generation_frames, dtype=torch.long, device=device
                )
                prediction = model(
                    list(batch["text"]),
                    generation_frames,
                    num_denoising_steps=args.diffusion_steps,
                    pad_mask=length_to_mask(lengths),
                    first_heading_angle=next_heading,
                    motion_mask=tail_mask,
                    observed_motion=tail_observed,
                    cfg_weight=(args.cfg_weight_text, args.cfg_weight_constraint),
                    crop_history_length=0,
                    text_feat=text_feat,
                    text_pad_mask=text_mask,
                    progress_bar=lambda values: values,
                )
                expected_windows = math.ceil(generation_frames / horizon)
                if client.controller.trace.accepted_commits != expected_windows + 1:
                    raise RuntimeError(
                        "persistent rollout did not commit one GT prefix and one state per window"
                    )
                if args.state_diagnostics_output:
                    state_diagnostics.extend(
                        {
                            "batch_start": start,
                            "sample_ids": [item["id"] for item in items],
                            **record,
                        }
                        for record in client.controller.state_diagnostics
                    )
                    client.controller.state_diagnostics.clear()
        torch.cuda.synchronize(device)
        latencies.extend([(time.perf_counter() - started) / size] * size)

        generated = model.motion_rep.inverse(prediction, is_normalized=True)
        target = model.motion_rep.inverse(tail_motion, is_normalized=True)
        history_geometry = model.motion_rep.inverse(init_history, is_normalized=True)
        generated_joints = generated["posed_joints"]
        target_joints = target["posed_joints"]
        generated_root = generated["root_positions"]
        target_root = target["root_positions"]
        generated_rotations = generated["local_rot_mats"]
        target_rotations = target["local_rot_mats"]

        for row, item in enumerate(items):
            for frame, joint in zip(item["goal_frames"], item["goal_joints"]):
                f, j = int(frame), int(joint)
                if j == model.skeleton.root_idx:
                    error = torch.linalg.vector_norm(
                        generated_root[row, f, [0, 2]] - target_root[row, f, [0, 2]]
                    )
                else:
                    error = torch.linalg.vector_norm(
                        generated_joints[row, f, j] - target_joints[row, f, j]
                    )
                constraint_errors.append(float(error * 100.0))
                window_constraint_errors[f // horizon].append(float(error * 100.0))
            root_distance = torch.linalg.vector_norm(
                generated_root[row] - target_root[row], dim=-1
            )
            joint_distance = torch.linalg.vector_norm(
                generated_joints[row] - target_joints[row], dim=-1
            ).mean(dim=-1)
            for window_index in range(num_windows):
                window_start = window_index * horizon
                window_end = min(window_start + horizon, generation_frames)
                window_root_errors[window_index].append(
                    float(root_distance[window_start:window_end].mean() * 100.0)
                )
                window_joint_errors[window_index].append(
                    float(joint_distance[window_start:window_end].mean() * 100.0)
                )
            root_ade_cm.append(float(root_distance.mean() * 100.0))
            root_fde_cm.append(float(root_distance[-1] * 100.0))
            pa_mpjpe_cm.append(
                float(_rigid_pa_mpjpe(generated_joints[row], target_joints[row]) * 100.0)
            )
            second_difference = (
                generated_joints[row, 2:]
                - 2.0 * generated_joints[row, 1:-1]
                + generated_joints[row, :-2]
            )
            jitter_m_per_frame2.append(
                float(torch.linalg.vector_norm(second_difference, dim=-1).mean())
            )
            angle_mae_deg.append(
                float(_angle_mae_degrees(generated_rotations[row], target_rotations[row]))
            )
            if args.artifacts_output:
                artifact_generated_joints.append(generated_joints[row].float().cpu().numpy())
                artifact_target_joints.append(target_joints[row].float().cpu().numpy())
                artifact_generated_root.append(generated_root[row].float().cpu().numpy())
                artifact_target_root.append(target_root[row].float().cpu().numpy())
                artifact_history_joints.append(
                    history_geometry["posed_joints"][row].float().cpu().numpy()
                )
                artifact_ids.append(item["id"])
                artifact_texts.append(item["text"])
                artifact_goal_frames.append(item["goal_frames"].numpy())
                artifact_goal_joints.append(item["goal_joints"].numpy())

        bad, total = _skate_counts(
            generated_joints, generated["foot_contacts"], foot_indices, fps
        )
        sliding += bad
        contacts += total
        speed = torch.linalg.vector_norm(
            generated_root[:, 1:] - generated_root[:, :-1], dim=-1
        ) * fps
        root_speeds.extend(float(value) for value in speed.flatten())
        boundary_indices = [
            frame - 1 for frame in range(horizon, generation_frames, horizon)
        ]
        for boundary in boundary_indices:
            boundary_root_speeds.extend(float(value) for value in speed[:, boundary])
        for row, item in enumerate(items):
            row_speed, frame = speed[row].max(dim=0)
            if float(row_speed) > max_root_speed_event["speed"]:
                max_root_speed_event = {
                    "speed": float(row_speed),
                    "sample_id": item["id"],
                    "frame": int(frame) + 1,
                }

    finite_values = (
        constraint_errors + root_speeds + latencies + root_ade_cm + root_fde_cm
        + pa_mpjpe_cm + jitter_m_per_frame2 + angle_mae_deg
    )
    def window_means(values: list[list[float]]) -> list[float | None]:
        return [float(np.mean(items)) if items else None for items in values]

    def slope(values: list[float | None], first_window: int) -> float | None:
        points = [
            (index + 1, value)
            for index, value in enumerate(values)
            if index + 1 >= first_window and value is not None
        ]
        if len(points) < 2:
            return None
        x = np.asarray([point[0] for point in points], dtype=np.float64)
        y = np.asarray([point[1] for point in points], dtype=np.float64)
        return float(np.polyfit(x, y, 1)[0])

    root_error_by_window = window_means(window_root_errors)
    joint_error_by_window = window_means(window_joint_errors)
    constraint_error_by_window = window_means(window_constraint_errors)
    paper_duration = math.isclose(total_frames / fps, 9.0)
    diagnostic_layers = [
        layer
        for record in state_diagnostics
        for layers in record["blocks"].values()
        for layer in layers
    ]
    mamba_norms = [
        layer["mamba_state_norm"]
        for layer in diagnostic_layers
        if layer["mamba_state_norm"] is not None
    ]
    mamba_drifts = [
        layer["mamba_state_drift_norm"]
        for layer in diagnostic_layers
        if layer["mamba_state_drift_norm"] is not None
    ]
    result = {
        "status": "PASS" if all(np.isfinite(finite_values)) else "FAIL",
        "metric_scope": (
            f"{'ARDY paper Table 5' if paper_duration else 'Extended-duration ARDY Table 5'} "
            f"generation protocol on {args.dataset_name}; "
            "learned official evaluator unavailable"
        ),
        "paper_reference": "ARDY Section 6.3 / Table 5, DOI 10.1145/3811284",
        "protocol_origin": "paper_exact_duration" if paper_duration else "duration_extension",
        "mode": args.mode,
        "model": args.model,
        "protocol": args.protocol,
        "samples": len(samples),
        "fps": fps,
        "total_frames": total_frames,
        "total_seconds": total_frames / fps,
        "selection_min_seconds": selection_min_frames / fps,
        "initial_history_frames": history_frames,
        "initial_history_seconds": history_frames / fps,
        "generated_frames": generation_frames,
        "generated_seconds": generation_frames / fps,
        "generation_horizon_frames": horizon,
        "generation_horizon_seconds": horizon / fps,
        "generation_windows": num_windows,
        "goals_per_sample": len(samples[0]["goal_frames"]),
        "goal_joint_pool": [
            model.skeleton.bone_order_names[index]
            for index in _goal_joint_indices(model.skeleton)
        ],
        "diffusion_steps": args.diffusion_steps,
        "cfg_weight_text": args.cfg_weight_text,
        "cfg_weight_constraint": args.cfg_weight_constraint,
        "memory_mode": "explicit_history" if args.mode == "original" else "persistent",
        "protocol_guard": (
            "native_explicit_history" if args.mode == "original"
            else "gt_prefix_primed_cache_only"
        ),
        "periodic_state_reset_windows": args.periodic_state_reset_windows,
        "mamba_output_decay_start_windows": args.mamba_output_decay_start_windows,
        "mamba_output_decay_half_life_windows": (
            args.mamba_output_decay_half_life_windows
        ),
        "mamba_output_decay_min_scale": args.mamba_output_decay_min_scale,
        "state_diagnostics_recorded": bool(args.state_diagnostics_output),
        "state_diagnostics_all_finite": (
            all(record["all_finite"] for record in state_diagnostics)
            if state_diagnostics else None
        ),
        "mamba_state_norm_max": max(mamba_norms) if mamba_norms else None,
        "mamba_state_drift_norm_max": max(mamba_drifts) if mamba_drifts else None,
        "r_precision": None,
        "fid": None,
        "constraint_error_cm_mean": float(np.mean(constraint_errors)),
        "constraint_error_cm_median": float(np.median(constraint_errors)),
        "mobench_root_ade_cm_mean": float(np.mean(root_ade_cm)),
        "mobench_root_fde_cm_mean": float(np.mean(root_fde_cm)),
        "mobench_pa_mpjpe_cm_mean": float(np.mean(pa_mpjpe_cm)),
        "mobench_jitter_m_per_frame2_mean": float(np.mean(jitter_m_per_frame2)),
        "mobench_angle_mae_deg_mean": float(np.mean(angle_mae_deg)),
        "root_ade_cm_mean_by_window": root_error_by_window,
        "joint_position_error_cm_mean_by_window": joint_error_by_window,
        "constraint_error_cm_mean_by_window": constraint_error_by_window,
        "post_attention_eviction_root_ade_slope_cm_per_window": slope(
            root_error_by_window, 5
        ),
        "post_attention_eviction_constraint_slope_cm_per_window": slope(
            constraint_error_by_window, 5
        ),
        "post_window5_root_ade_slope_cm_per_window": slope(root_error_by_window, 6),
        "post_window5_constraint_slope_cm_per_window": slope(
            constraint_error_by_window, 6
        ),
        "skate_percent_proxy": 100.0 * sliding / max(contacts, 1),
        "latency_seconds_per_sample": float(np.mean(latencies)),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "mean_root_speed": float(np.mean(root_speeds)),
        "max_root_speed": float(np.max(root_speeds)),
        "mean_window_boundary_root_speed": float(np.mean(boundary_root_speeds)),
        "max_window_boundary_root_speed": float(np.max(boundary_root_speeds)),
        "max_root_speed_sample_id": max_root_speed_event["sample_id"],
        "max_root_speed_frame": max_root_speed_event["frame"],
        "active_parameters": sum(
            parameter.numel()
            for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters())
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "normalization_stats": str(Path(args.stats_dir).resolve()),
        "sample_ids": [item["id"] for item in samples],
        "goal_frames_by_sample": [item["goal_frames"].tolist() for item in samples],
        "goal_joints_by_sample": [item["goal_joints"].tolist() for item in samples],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.state_diagnostics_output:
        diagnostics_path = Path(args.state_diagnostics_output)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            "".join(json.dumps(record) + "\n" for record in state_diagnostics)
        )
    if args.artifacts_output:
        artifact_path = Path(args.artifacts_output)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            generated_joints=np.stack(artifact_generated_joints),
            target_joints=np.stack(artifact_target_joints),
            generated_root=np.stack(artifact_generated_root),
            target_root=np.stack(artifact_target_root),
            initial_history_joints=np.stack(artifact_history_joints),
            sample_ids=np.asarray(artifact_ids),
            texts=np.asarray(artifact_texts),
            goal_frames=np.stack(artifact_goal_frames),
            goal_joints=np.stack(artifact_goal_joints),
        )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
