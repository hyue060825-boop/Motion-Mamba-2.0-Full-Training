"""Evaluate reproducible Table 4/5 proxy metrics for trained ARDY adapters.

This checkout does not include the paper's HumanML3D retrieval/FID evaluator.
The script therefore reports only directly measurable geometry, constraint,
latency, and memory metrics and labels the missing learned metrics explicitly.
"""

from __future__ import annotations

import argparse
import json
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
from train_humanml3d_ardy_tiny import _replace_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "minimal", "hybrid"], required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Required for Minimal/Hybrid; Original uses the released ARDY weights.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--protocol", choices=["table4", "table5_in", "table5_out"], required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cfg-weight-text", type=float, default=2.0)
    parser.add_argument("--cfg-weight-constraint", type=float, default=2.0)
    parser.add_argument(
        "--memory-mode",
        choices=["persistent", "explicit_history"],
        default="persistent",
        help="Disable cross-window commits for a recentered-history diagnostic.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help="Explicit autoregressive history; defaults to 0 for Table 4 and 40 for Table 5.",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _select_indices(dataset: HumanML3DArdyDataset, frames: int, count: int, seed: int) -> list[int]:
    eligible = []
    for index, record in enumerate(dataset.records):
        motion = np.load(record["feature_path"], mmap_mode="r")
        if len(motion) >= frames:
            eligible.append(index)
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} records have at least {frames} frames")
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.choice(eligible, size=count, replace=False)]


def _joint_from_mask(mask: torch.Tensor) -> tuple[int, int]:
    positions = torch.nonzero(mask[:, 5:83].any(dim=-1), as_tuple=False).flatten()
    if len(positions) != 1:
        raise RuntimeError(f"expected one constrained frame, got {positions.tolist()}")
    frame = int(positions[0])
    columns = torch.nonzero(mask[frame, 5:83], as_tuple=False).flatten()
    return frame, int(columns[0] // 3) + 1


def _skate_counts(
    posed: torch.Tensor,
    foot_contacts: torch.Tensor,
    foot_indices: list[int],
    fps: int,
) -> tuple[int, int]:
    feet = posed[:, :, foot_indices]
    horizontal_speed = torch.linalg.vector_norm(
        feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]], dim=-1
    ) * fps
    contact = foot_contacts[:, 1:].bool()
    return int(((horizontal_speed > 0.15) & contact).sum()), contact.numel()


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch size must be positive")
    if args.mode != "original" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for Minimal/Hybrid evaluation")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    from ardy.model import load_model
    from ardy.motion_rep.tools import length_to_mask
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model("core", device=str(device), text_encoder=False)
    dataset_stats_dir = args.stats_dir
    if args.mode != "original":
        _replace_stats(model, args.stats_dir)
    else:
        # Released ARDY weights were trained with the bundled Rigplay stats.
        # Re-encode HumanML3D physical motion with those stats for a valid
        # released-checkpoint reference instead of mutating its normalization.
        dataset_stats_dir = model.motion_rep.stats.folder
    raw = model.denoiser.model
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
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
        if args.memory_mode == "explicit_history":
            client.controller.uninstall()
    raw.root_model.eval()
    raw.body_model.eval()

    frames = 40 if args.protocol == "table4" else 80
    history_frames = args.history_frames
    if history_frames is None:
        history_frames = 0 if args.protocol == "table4" else 40
    out_of_horizon = args.protocol == "table5_out"
    dataset = HumanML3DArdyDataset(
        args.manifest,
        dataset_stats_dir,
        text_root=args.text_root,
        frames=frames,
        horizon=40,
        condition_dropout=0.0,
        seed=args.seed,
        text_cache_dir=args.text_cache,
        history_probability=0.0,
    )
    indices = _select_indices(dataset, frames, args.samples, args.seed)
    samples = [
        dataset.sample(
            index,
            args.seed + 1000003 * offset,
            control="joint_position",
            out_of_horizon=out_of_horizon,
        )
        for offset, index in enumerate(indices)
    ]

    constraint_errors = []
    root_speeds = []
    boundary_root_speeds = []
    max_root_speed_event = {"speed": float("-inf"), "sample_id": None, "frame": None}
    latencies = []
    sliding = 0
    contacts = 0
    peak_memory = 0.0
    foot_indices = list(model.skeleton.foot_joint_idx)
    fps = int(model.motion_rep.fps)
    torch.cuda.reset_peak_memory_stats(device)

    for start in range(0, len(samples), args.batch_size):
        items = samples[start:start + args.batch_size]
        batch = default_collate(items)
        size = len(items)
        lengths = torch.full((size,), frames, dtype=torch.long, device=device)
        pad_mask = length_to_mask(lengths)
        motion_mask = batch["motion_mask"].to(device)
        observed = batch["observed_motion"].to(device)
        text_feat = batch["text_feature"].to(device)[:, None]
        text_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = client(
                list(batch["text"]),
                frames,
                num_denoising_steps=args.diffusion_steps,
                pad_mask=pad_mask,
                first_heading_angle=torch.zeros(size, device=device),
                motion_mask=motion_mask,
                observed_motion=observed,
                cfg_weight=(args.cfg_weight_text, args.cfg_weight_constraint),
                crop_history_length=history_frames,
                text_feat=text_feat,
                text_pad_mask=text_mask,
                progress_bar=lambda values: values,
            )
        torch.cuda.synchronize(device)
        latencies.extend([(time.perf_counter() - started) / size] * size)
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device) / 2**20)

        generated = model.motion_rep.inverse(prediction, is_normalized=True)
        target = model.motion_rep.inverse(batch["motion"].to(device), is_normalized=True)
        generated_joints = generated["posed_joints"]
        target_joints = target["posed_joints"]
        for row, item in enumerate(items):
            frame, joint = _joint_from_mask(item["motion_mask"])
            error_cm = torch.linalg.vector_norm(
                generated_joints[row, frame, joint] - target_joints[row, frame, joint]
            ) * 100.0
            constraint_errors.append(float(error_cm))
        bad, total = _skate_counts(
            generated_joints, generated["foot_contacts"], foot_indices, fps
        )
        sliding += bad
        contacts += total
        root = generated["root_positions"]
        speed = torch.linalg.vector_norm(root[:, 1:] - root[:, :-1], dim=-1) * fps
        root_speeds.extend(float(value) for value in speed.flatten())
        if frames > model.gen_horizon_len:
            boundary_root_speeds.extend(
                float(value) for value in speed[:, model.gen_horizon_len - 1]
            )
        for row, item in enumerate(items):
            row_speed, frame = speed[row].max(dim=0)
            if float(row_speed) > max_root_speed_event["speed"]:
                max_root_speed_event = {
                    "speed": float(row_speed),
                    "sample_id": item["id"],
                    "frame": int(frame) + 1,
                }

    result = {
        "status": "PASS" if all(np.isfinite(constraint_errors + root_speeds + latencies)) else "FAIL",
        "metric_scope": "ARDY Table 4/5 proxy; learned HumanML3D evaluator unavailable",
        "mode": args.mode,
        "protocol": args.protocol,
        "samples": len(samples),
        "frames": frames,
        "diffusion_steps": args.diffusion_steps,
        "cfg_weight_text": args.cfg_weight_text,
        "cfg_weight_constraint": args.cfg_weight_constraint,
        "history_frames": history_frames,
        "memory_mode": args.memory_mode,
        "r_precision": None,
        "fid": None,
        "constraint_error_cm_mean": float(np.mean(constraint_errors)),
        "constraint_error_cm_median": float(np.median(constraint_errors)),
        "skate_percent_proxy": 100.0 * sliding / max(contacts, 1),
        "latency_seconds_per_sample": float(np.mean(latencies)),
        "peak_gpu_memory_mib": peak_memory,
        "mean_root_speed": float(np.mean(root_speeds)),
        "max_root_speed": float(np.max(root_speeds)),
        "mean_window_boundary_root_speed": (
            float(np.mean(boundary_root_speeds)) if boundary_root_speeds else None
        ),
        "max_window_boundary_root_speed": (
            float(np.max(boundary_root_speeds)) if boundary_root_speeds else None
        ),
        "max_root_speed_sample_id": max_root_speed_event["sample_id"],
        "max_root_speed_frame": max_root_speed_event["frame"],
        "active_parameters": sum(
            parameter.numel()
            for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters())
        ),
        "checkpoint": (
            "released_ardy_core"
            if args.checkpoint is None
            else str(Path(args.checkpoint).resolve())
        ),
        "normalization_stats": str(Path(dataset_stats_dir).resolve()),
        "sample_ids": [item["id"] for item in samples],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
