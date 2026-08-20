"""Real-data/control tiny-overfit gate for Minimal or Hybrid ARDY mixers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate

REPO = Path(__file__).resolve().parents[1]
ARDY_ROOT = REPO.parents[0] / "ardy"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ARDY_ROOT))
for path in (
    os.environ.get("MAMBA_REPO", "/data3/zts/packages/state-spaces-mamba"),
    os.environ.get("MAMBA_TRITON_OVERLAY", "/data3/zts/packages/mamba_op_triton"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from ardy_data import HumanML3DArdyDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", required=True)
    parser.add_argument("--text-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["minimal", "hybrid"], required=True)
    parser.add_argument("--train-scope", choices=["mixer", "denoiser"], default="denoiser")
    parser.add_argument("--phase", choices=["tiny", "pilot"], default="tiny")
    parser.add_argument("--dev-manifest", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tiny-samples", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--history-probability", type=float, default=0.0)
    parser.add_argument("--constraint-dropout", type=float, default=0.1)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--root-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--fk-constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--foot-skate-loss-weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def _replace_stats(model, stats_dir: str):
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep

    old = model.motion_rep
    rep = ArdyMotionRep(old.skeleton, old.fps, stats_path=stats_dir)
    rep.skeleton = old.skeleton
    model.motion_rep = rep
    raw = model.denoiser.model
    raw.motion_rep = rep
    model.autoencoder._motion_rep = rep
    for converter in (model.hybrid, raw.hybrid):
        converter._motion_rep = rep
        converter.__dict__.pop("motion_rep", None)
    return rep


def _hashed_text(texts: list[str], dim: int, device: torch.device) -> torch.Tensor:
    values = []
    for caption in texts:
        seed = int.from_bytes(hashlib.sha256(caption.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(dim).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        values.append(torch.from_numpy(vector))
    return torch.stack(values)[:, None].to(device)


def _build_batch(
    model,
    batch,
    device: torch.device,
    fixed_noise: bool,
    boundary_loss_weight: float = 0.0,
    root_velocity_loss_weight: float = 0.0,
    constraint_loss_weight: float = 0.0,
    fk_constraint_loss_weight: float = 0.0,
    foot_skate_loss_weight: float = 0.0,
    sequential_cache: bool = False,
):
    from ardy.model.ardy_model import get_three_mask_from_len

    motion = batch["motion"].to(device)
    observed = batch["observed_motion"].to(device)
    motion_mask = batch["motion_mask"].to(device)
    batch_size, frames = motion.shape[:2]
    if frames % model.hybrid.num_frames_per_token:
        raise ValueError("frame count must be divisible by the tokenizer patch size")
    lengths = torch.full((batch_size,), frames, dtype=torch.long, device=device)
    pad = torch.ones((batch_size, frames), dtype=torch.bool, device=device)
    with torch.no_grad():
        clean, _ = model.hybrid.get_hybrid_motion_from_explicit(motion, lengths, pad)
    history_len = batch.get("history_frames", torch.zeros(batch_size, dtype=torch.long)).to(device)
    generation_len = torch.full((batch_size,), 40, dtype=torch.long, device=device)
    future_len = frames - history_len - generation_len
    if bool((future_len < 0).any()):
        raise ValueError("history + generation exceeds the training window")
    history_mask, generation_mask, future_mask = get_three_mask_from_len(
        history_len, generation_len, future_len, frames, device
    )
    history_token_mask, generation_token_mask, future_token_mask = model.hybrid.convert_frame_mask_to_token_mask(
        history_mask, generation_mask, future_mask, motion_mask
    )
    if "text_feature" in batch:
        text = batch["text_feature"].to(device)[:, None]
    else:
        text = _hashed_text(list(batch["text"]), model.denoiser.model.llm_shape[-1], device)
    text_dropped = batch.get(
        "text_dropped", torch.zeros(batch_size, dtype=torch.bool)
    ).to(device)
    text_mask = (~text_dropped)[:, None].expand(text.shape[:2])
    # Heading is an angle in the physical [cos, sin] representation; compute
    # it before normalization rather than applying atan2 to z-scored values.
    physical_motion = model.motion_rep.unnormalize(motion)
    heading = torch.atan2(physical_motion[:, 0, 4], physical_motion[:, 0, 3])
    with torch.no_grad():
        target_geometry = model.motion_rep.inverse(motion, is_normalized=True)
    if fixed_noise:
        timestep = torch.full((batch_size,), model.diffusion.num_base_steps // 2, dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(901)
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    else:
        timestep = torch.randint(0, model.diffusion.num_base_steps, (batch_size,), device=device)
        noise = torch.randn_like(clean)
    noisy = model.diffusion.q_sample(clean, timestep, noise)
    x = torch.zeros_like(clean)
    x[history_token_mask] = clean[history_token_mask]
    x[generation_token_mask] = noisy[generation_token_mask]
    return {
        "x": x, "target": clean,
        "history_len": history_len, "generation_len": generation_len, "future_len": future_len,
        "history_mask": history_mask, "generation_mask": generation_mask, "future_mask": future_mask,
        "history_token_mask": history_token_mask, "generation_token_mask": generation_token_mask,
        "future_token_mask": future_token_mask, "text": text, "text_mask": text_mask,
        "timestep": timestep, "heading": heading, "motion_mask": motion_mask,
        "observed": observed, "boundary_loss_weight": boundary_loss_weight,
        "root_velocity_loss_weight": root_velocity_loss_weight,
        "constraint_loss_weight": constraint_loss_weight,
        "fk_constraint_loss_weight": fk_constraint_loss_weight,
        "foot_skate_loss_weight": foot_skate_loss_weight,
        "sequential_cache": sequential_cache,
        "motion": motion,
        "lengths": lengths,
        "pad_mask": pad,
        "hybrid_converter": model.hybrid,
        "target_joints": target_geometry["posed_joints"],
        "target_foot_contacts": target_geometry["foot_contacts"],
        "joint_position_mask": motion_mask[:, :, 5:83].reshape(
            batch_size, frames, 26, 3
        ).any(dim=-1) & generation_mask[:, :, None],
        "foot_indices": list(model.skeleton.foot_joint_idx),
    }


def _unwrap_block(block):
    return getattr(block, "module", block)


def _unwrap_raw(raw):
    return getattr(raw, "module", raw)


def _denoiser_aux_state(raw) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in _unwrap_raw(raw).state_dict().items()
        if not key.startswith(("root_model.", "body_model."))
    }


def _load_denoiser_weights(raw, checkpoint: dict) -> None:
    model = _unwrap_raw(raw)
    _unwrap_block(model.root_model).load_state_dict(checkpoint["root_model"], strict=True)
    _unwrap_block(model.body_model).load_state_dict(checkpoint["body_model"], strict=True)
    auxiliary = checkpoint.get("denoiser_aux")
    if auxiliary is not None:
        current = model.state_dict()
        unknown = set(auxiliary) - set(current)
        if unknown:
            raise ValueError(f"checkpoint has unknown denoiser_aux keys: {sorted(unknown)}")
        with torch.no_grad():
            for key, value in auxiliary.items():
                current[key].copy_(value.to(device=current[key].device, dtype=current[key].dtype))


def _prime_teacher_history(raw, fixed) -> None:
    """Commit recentered clean history before the noisy second-window read."""
    selection = fixed["history_token_mask"]
    counts = selection.sum(dim=1)
    if int(counts.min()) == 0 or not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("sequential cache batches require equal non-empty history")
    prime_x = torch.zeros_like(fixed["target"])
    prime_x[selection] = fixed["target"][selection]
    model = _unwrap_raw(raw)
    root = _unwrap_block(model.root_model)
    body = _unwrap_block(model.body_model)
    root.prepare_accepted_commit(selection, chunk_id=0, detach=True)
    body.prepare_accepted_commit(selection, chunk_id=0, detach=True)
    # Persistent projected K/V use the fp32 attention projection dtype. This
    # commit may be called under the outer bf16 training/evaluation autocast.
    with torch.no_grad(), torch.autocast(
        device_type=prime_x.device.type,
        enabled=False,
    ):
        raw(
            prime_x, fixed["history_len"], fixed["generation_len"], fixed["future_len"],
            fixed["history_mask"], fixed["generation_mask"], fixed["future_mask"],
            fixed["history_token_mask"], fixed["generation_token_mask"],
            fixed["future_token_mask"], fixed["text"], fixed["text_mask"],
            torch.zeros_like(fixed["timestep"]), fixed["heading"],
            fixed["motion_mask"], fixed["observed"],
        )
    for name, block in (("root", root), ("body", body)):
        cache = block._sequence_state.layers[0].attention_cache
        if cache.chunk_ids != (0,):
            raise RuntimeError(f"{name} teacher history was not committed exactly once")


def _loss(raw, client, fixed, return_components: bool = False):
    client.reset_memory()
    if fixed.get("sequential_cache", False):
        _prime_teacher_history(raw, fixed)
    prediction = raw(
        fixed["x"], fixed["history_len"], fixed["generation_len"], fixed["future_len"],
        fixed["history_mask"], fixed["generation_mask"], fixed["future_mask"],
        fixed["history_token_mask"], fixed["generation_token_mask"], fixed["future_token_mask"],
        fixed["text"], fixed["text_mask"], fixed["timestep"], fixed["heading"],
        fixed["motion_mask"], fixed["observed"],
    )
    selection = fixed["generation_token_mask"]
    predicted = prediction[selection].float()
    target = fixed["target"][selection].float()
    model = _unwrap_raw(raw)
    root_dim = model.nframe_root_dim
    root_loss = F.mse_loss(predicted[:, :root_dim], target[:, :root_dim])
    body_loss = F.mse_loss(predicted[:, root_dim:], target[:, root_dim:])
    rows = torch.arange(prediction.shape[0], device=prediction.device)
    first_generation = selection.long().argmax(dim=1)
    predicted_boundary = prediction[rows, first_generation][:, [0, 2]].float()
    target_boundary = fixed["target"][rows, first_generation][:, [0, 2]].float()
    boundary_loss = F.mse_loss(predicted_boundary, target_boundary)
    nfpt = model.num_frames_per_token
    root_pred = prediction[:, :, :root_dim].float().reshape(
        prediction.shape[0], prediction.shape[1] * nfpt, -1
    )
    root_target = fixed["target"][:, :, :root_dim].float().reshape(
        prediction.shape[0], prediction.shape[1] * nfpt, -1
    )
    # Include the history-to-generation transition. This is the actual
    # autoregressive window boundary whose velocity must remain continuous.
    velocity_mask = fixed["generation_mask"][:, 1:] & (
        fixed["history_mask"][:, :-1] | fixed["generation_mask"][:, :-1]
    )
    predicted_velocity = root_pred[:, 1:, [0, 2]] - root_pred[:, :-1, [0, 2]]
    target_velocity = root_target[:, 1:, [0, 2]] - root_target[:, :-1, [0, 2]]
    velocity_loss = F.mse_loss(
        predicted_velocity[velocity_mask], target_velocity[velocity_mask]
    )
    # Future constraints condition the current window but the ARDY denoiser
    # deliberately zeroes future outputs. They cannot be direct loss targets.
    constraint_mask = (
        fixed["motion_mask"].bool()
        & fixed["generation_mask"][:, :, None]
    )
    if constraint_mask.any():
        explicit_prediction = fixed["hybrid_converter"].get_explicit_motion_from_hybrid(
            prediction, fixed["pad_mask"], fixed["lengths"]
        )
        constraint_loss = F.mse_loss(
            explicit_prediction[constraint_mask].float(),
            fixed["motion"][constraint_mask].float(),
        )
    else:
        constraint_loss = prediction.new_zeros(())
        explicit_prediction = None
    needs_geometry = (
        fixed["fk_constraint_loss_weight"] > 0
        or fixed["foot_skate_loss_weight"] > 0
    )
    if needs_geometry:
        if explicit_prediction is None:
            explicit_prediction = fixed["hybrid_converter"].get_explicit_motion_from_hybrid(
                prediction, fixed["pad_mask"], fixed["lengths"]
            )
        generated_geometry = model.motion_rep.inverse(
            explicit_prediction, is_normalized=True
        )
        generated_joints = generated_geometry["posed_joints"]
        joint_mask = (
            fixed["joint_position_mask"]
            & fixed["generation_mask"][:, :, None]
        )
        if joint_mask.any():
            fk_constraint_loss = F.mse_loss(
                generated_joints[:, :, 1:][joint_mask].float(),
                fixed["target_joints"][:, :, 1:][joint_mask].float(),
            )
        else:
            fk_constraint_loss = prediction.new_zeros(())
        feet = generated_joints[:, :, fixed["foot_indices"]]
        horizontal_velocity = feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]]
        contact = (
            fixed["target_foot_contacts"][:, 1:].bool()
            & velocity_mask[:, :, None]
        )
        if contact.any():
            foot_skate_loss = horizontal_velocity[contact].square().mean()
        else:
            foot_skate_loss = prediction.new_zeros(())
    else:
        fk_constraint_loss = prediction.new_zeros(())
        foot_skate_loss = prediction.new_zeros(())
    total = (
        root_loss + body_loss
        + fixed["boundary_loss_weight"] * boundary_loss
        + fixed["root_velocity_loss_weight"] * velocity_loss
        + fixed["constraint_loss_weight"] * constraint_loss
        + fixed["fk_constraint_loss_weight"] * fk_constraint_loss
        + fixed["foot_skate_loss_weight"] * foot_skate_loss
    )
    if return_components:
        return (
            total, root_loss, body_loss, boundary_loss, velocity_loss,
            constraint_loss, fk_constraint_loss, foot_skate_loss,
        )
    return total


@torch.no_grad()
def _evaluate(raw, client, fixed):
    model = _unwrap_raw(raw)
    root_training = model.root_model.training
    body_training = model.body_model.training
    model.root_model.eval()
    model.body_model.eval()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return float(_loss(raw, client, fixed))
    finally:
        model.root_model.train(root_training)
        model.body_model.train(body_training)


def _evaluate_many(raw, client, batches) -> float:
    return float(np.mean([_evaluate(raw, client, batch) for batch in batches]))


def _save_resume_checkpoint(
    path: Path,
    *,
    args,
    raw,
    optimizer,
    step: int,
    losses: list[float],
    initial_eval: float,
    all_finite: bool,
    data_generator: torch.Generator,
    elapsed_seconds: float,
    device: torch.device,
) -> None:
    checkpoint = {
        "format_version": 2,
        "mode": args.mode,
        "phase": args.phase,
        "step": step,
        "root_model": _unwrap_block(_unwrap_raw(raw).root_model).state_dict(),
        "body_model": _unwrap_block(_unwrap_raw(raw).body_model).state_dict(),
        "denoiser_aux": _denoiser_aux_state(raw),
        "optimizer": optimizer.state_dict(),
        "losses": losses,
        "initial_eval_loss": initial_eval,
        "all_gradients_finite": all_finite,
        "data_generator_state": data_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "elapsed_seconds": elapsed_seconds,
        "config": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model("core", device=str(device), text_encoder=False)
    _replace_stats(model, args.stats_dir)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    client = install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode=args.mode,
            attention_backend="sdpa",
            window_size_chunks=4,
            num_sink_tokens=4,
            token_state_interaction="naive_gated_add",
        ),
    )
    raw = model.denoiser.model
    if args.train_scope == "denoiser":
        for parameter in raw.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters()):
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    raw.root_model.train()
    raw.body_model.train()
    model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))
    dataset = HumanML3DArdyDataset(
        args.manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=40,
        condition_dropout=args.constraint_dropout, seed=args.seed, text_cache_dir=args.text_cache,
        history_probability=args.history_probability,
        text_dropout=args.text_dropout,
    )
    data_generator = torch.Generator().manual_seed(args.seed + 555)
    if args.phase == "tiny":
        training_data = Subset(dataset, range(min(args.tiny_samples, len(dataset))))
        loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=False)
        batch = next(iter(loader))
    else:
        batch = default_collate([dataset[index] for index in range(args.batch_size)])
    fixed = _build_batch(
        model, batch, device, fixed_noise=True,
        boundary_loss_weight=args.boundary_loss_weight,
        root_velocity_loss_weight=args.root_velocity_loss_weight,
        constraint_loss_weight=args.constraint_loss_weight,
        fk_constraint_loss_weight=args.fk_constraint_loss_weight,
        foot_skate_loss_weight=args.foot_skate_loss_weight,
    )
    dev_fixed = []
    if args.dev_manifest:
        dev = HumanML3DArdyDataset(
            args.dev_manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=40,
            condition_dropout=0.0, seed=args.seed + 10000, text_cache_dir=args.text_cache,
            history_probability=args.history_probability,
            text_dropout=0.0,
        )
        for batch_index in range(args.eval_batches):
            start = batch_index * args.batch_size
            samples = [dev[(start + offset) % len(dev)] for offset in range(args.batch_size)]
            dev_fixed.append(
                _build_batch(
                    model, default_collate(samples), device, fixed_noise=True,
                    boundary_loss_weight=args.boundary_loss_weight,
                    root_velocity_loss_weight=args.root_velocity_loss_weight,
                    constraint_loss_weight=args.constraint_loss_weight,
                    fk_constraint_loss_weight=args.fk_constraint_loss_weight,
                    foot_skate_loss_weight=args.foot_skate_loss_weight,
                )
            )
    eval_batches = dev_fixed or [fixed]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    start_step = 0
    losses = []
    all_finite = True
    elapsed_before = 0.0
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("mode") != args.mode:
            raise ValueError("initial checkpoint mode does not match requested run")
        _load_denoiser_weights(raw, initial)
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("mode") != args.mode or resume.get("phase") != args.phase:
            raise ValueError("resume checkpoint mode/phase does not match requested run")
        _load_denoiser_weights(raw, resume)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])
        losses = list(resume["losses"])
        all_finite = bool(resume["all_gradients_finite"])
        data_generator.set_state(resume["data_generator_state"].cpu())
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device)
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))
        initial_eval = float(resume["initial_eval_loss"])
        if args.steps <= start_step:
            raise ValueError(f"--steps must exceed resumed step {start_step}")
    else:
        initial_eval = _evaluate_many(raw, client, eval_batches)
        if log_path.exists():
            raise FileExistsError(f"refusing to overwrite existing log without --resume: {log_path}")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        if args.phase == "pilot":
            indices = torch.randint(len(dataset), (args.batch_size,), generator=data_generator).tolist()
            seeds = torch.randint(
                0, 2**63 - 1, (args.batch_size,), generator=data_generator
            ).tolist()
            batch = default_collate(
                [dataset.sample(index, seed) for index, seed in zip(indices, seeds)]
            )
            current = _build_batch(
                model, batch, device, fixed_noise=False,
                boundary_loss_weight=args.boundary_loss_weight,
                root_velocity_loss_weight=args.root_velocity_loss_weight,
                constraint_loss_weight=args.constraint_loss_weight,
                fk_constraint_loss_weight=args.fk_constraint_loss_weight,
                foot_skate_loss_weight=args.foot_skate_loss_weight,
            )
        else:
            current = fixed
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                loss, root_loss, body_loss, boundary_loss, velocity_loss,
                constraint_loss, fk_constraint_loss, foot_skate_loss,
            ) = _loss(raw, client, current, return_components=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(grad).all() for grad in gradients)
        all_finite = all_finite and finite
        if not finite:
            raise RuntimeError(f"missing/non-finite gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss),
                "root_loss": float(root_loss),
                "body_loss": float(body_loss),
                "boundary_loss": float(boundary_loss),
                "root_velocity_loss": float(velocity_loss),
                "constraint_loss": float(constraint_loss),
                "fk_constraint_loss": float(fk_constraint_loss),
                "foot_skate_loss": float(foot_skate_loss),
                "grad_norm": float(grad_norm),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            eval_loss = _evaluate_many(raw, client, eval_batches)
            record = {"step": step, "eval_loss": eval_loss}
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if args.save_every and step % args.save_every == 0:
            _save_resume_checkpoint(
                output / f"checkpoint_step_{step:06d}.pt",
                args=args, raw=raw, optimizer=optimizer, step=step,
                losses=losses, initial_eval=initial_eval, all_finite=all_finite,
                data_generator=data_generator,
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
                device=device,
            )
    final_eval = _evaluate_many(raw, client, eval_batches)
    first_window = float(np.mean(losses[:min(50, len(losses))]))
    last_window = float(np.mean(losses[-min(50, len(losses)):]))
    passed = (
        final_eval < 0.5 * initial_eval
        if args.phase == "tiny"
        else all_finite and last_window < first_window and final_eval < initial_eval
    )
    metrics = {
        "status": "PASS" if passed else "FAIL",
        "mode": args.mode,
        "phase": args.phase,
        "steps": args.steps,
        "samples": len(batch["id"]),
        "ids": list(batch["id"]),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": 1.0 - losses[-1] / losses[0],
        "first_50_mean_loss": first_window,
        "last_50_mean_loss": last_window,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "all_gradients_finite": all_finite,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "constraint_elements": [int(value) for value in fixed["motion_mask"].sum((1, 2))],
        "out_of_horizon": [bool(value) for value in batch["out_of_horizon"]],
        "history_frames": [int(value) for value in batch["history_frames"]],
        "text_backend": "llm2vec_cache" if args.text_cache else "deterministic_hash_smoke_only",
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    checkpoint = {
        "mode": args.mode,
        "root_model": _unwrap_block(_unwrap_raw(raw).root_model).state_dict(),
        "body_model": _unwrap_block(_unwrap_raw(raw).body_model).state_dict(),
        "denoiser_aux": _denoiser_aux_state(raw),
        "metrics": metrics,
    }
    torch.save(checkpoint, output / f"{args.mode}_humanml3d_{args.phase}.pt")
    if not args.save_every or args.steps % args.save_every != 0:
        _save_resume_checkpoint(
            output / f"checkpoint_step_{args.steps:06d}.pt",
            args=args, raw=raw, optimizer=optimizer, step=args.steps,
            losses=losses, initial_eval=initial_eval, all_finite=all_finite,
            data_generator=data_generator, elapsed_seconds=metrics["elapsed_seconds"],
            device=device,
        )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if metrics["status"] != "PASS":
        raise RuntimeError(f"{args.phase} training gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
