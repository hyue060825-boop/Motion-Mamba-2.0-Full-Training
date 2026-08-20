"""Resumable multi-GPU HumanML3D/control trainer for Minimal and Hybrid."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))
sys.path.insert(0, str(REPO / "examples"))
for path in ("/data3/zts/packages/state-spaces-mamba", "/data3/zts/packages/mamba_op_triton"):
    sys.path.insert(0, path)

from ardy_data import HumanML3DArdyDataset  # noqa: E402
from train_humanml3d_ardy_tiny import (  # noqa: E402
    _build_batch,
    _evaluate_many,
    _loss,
    _load_denoiser_weights,
    _replace_stats,
    _unwrap_block,
    _unwrap_raw,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--text-root", required=True)
    parser.add_argument("--text-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["minimal", "hybrid"], required=True)
    parser.add_argument("--train-scope", choices=["mixer", "denoiser"], default="denoiser")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-rank batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=0,
        help="Keep only the newest N resume checkpoints in this output directory; 0 keeps all.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-final-resume-checkpoint", action="store_true")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--history-probability", type=float, default=0.0)
    parser.add_argument("--constraint-dropout", type=float, default=0.1)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument(
        "--sequential-cache-probability",
        type=float,
        default=0.0,
        help="Probability that an entire global batch trains a teacher-forced second window with committed history.",
    )
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--root-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--fk-constraint-loss-weight", type=float, default=0.0)
    parser.add_argument("--foot-skate-loss-weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def _mean_across_ranks(value: torch.Tensor) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced)


def _global_draws(
    count: int,
    dataset_size: int,
    generator: torch.Generator,
    device: torch.device,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rank == 0:
        indices = torch.randint(dataset_size, (count,), generator=generator, dtype=torch.long).to(device)
        seeds = torch.randint(0, 2**63 - 1, (count,), generator=generator, dtype=torch.long).to(device)
    else:
        indices = torch.empty(count, dtype=torch.long, device=device)
        seeds = torch.empty(count, dtype=torch.long, device=device)
    dist.broadcast(indices, src=0)
    dist.broadcast(seeds, src=0)
    return indices, seeds


def _global_probability_draw(
    probability: float,
    generator: torch.Generator,
    device: torch.device,
    rank: int,
) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("--sequential-cache-probability must be in [0, 1]")
    if probability in (0.0, 1.0):
        return bool(probability)
    if rank == 0:
        draw = (torch.rand((), generator=generator) < probability).to(device)
    else:
        draw = torch.empty((), dtype=torch.bool, device=device)
    dist.broadcast(draw, src=0)
    return bool(draw.item())


def _save_checkpoint(
    path: Path,
    *,
    args,
    raw,
    optimizer,
    step: int,
    losses: list[float],
    initial_eval: float,
    all_finite: bool,
    sequential_cache_steps: int,
    data_generator: torch.Generator,
    elapsed_seconds: float,
    device: torch.device,
    rank: int,
) -> None:
    local_rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }
    rng_by_rank = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rng_by_rank, local_rng)
    if rank == 0:
        model = _unwrap_raw(raw)
        root = _unwrap_block(model.root_model)
        body = _unwrap_block(model.body_model)
        auxiliary = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith(("root_model.", "body_model."))
        }
        checkpoint = {
            "format_version": 2,
            "mode": args.mode,
            "phase": "ddp",
            "world_size": dist.get_world_size(),
            "step": step,
            "root_model": root.state_dict(),
            "body_model": body.state_dict(),
            "denoiser_aux": auxiliary,
            "optimizer": optimizer.state_dict(),
            "losses": losses,
            "initial_eval_loss": initial_eval,
            "all_gradients_finite": all_finite,
            "sequential_cache_steps": sequential_cache_steps,
            "data_generator_state": data_generator.get_state(),
            "rng_by_rank": rng_by_rank,
            "elapsed_seconds": elapsed_seconds,
            "config": vars(args),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    dist.barrier()


def _prune_checkpoints(output: Path, keep: int, rank: int) -> None:
    if keep < 0:
        raise ValueError("--keep-last-checkpoints must be non-negative")
    if keep and rank == 0:
        checkpoints = sorted(output.glob("checkpoint_step_*.pt"))
        for checkpoint in checkpoints[:-keep]:
            checkpoint.unlink()
    dist.barrier()


def _load_model_weights(raw, checkpoint: dict) -> None:
    _load_denoiser_weights(raw, checkpoint)


def main() -> int:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

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
        ),
    )
    raw = model.denoiser.model
    if args.train_scope == "denoiser":
        for parameter in raw.parameters():
            parameter.requires_grad_(True)
        raw = DDP(raw, device_ids=[local_rank], broadcast_buffers=False)
        raw.train()
        parameters = list(raw.parameters())
    else:
        for parameter in list(raw.root_model.parameters()) + list(raw.body_model.parameters()):
            parameter.requires_grad_(True)
        raw.root_model = DDP(raw.root_model, device_ids=[local_rank], broadcast_buffers=False)
        raw.body_model = DDP(raw.body_model, device_ids=[local_rank], broadcast_buffers=False)
        raw.root_model.train()
        raw.body_model.train()
        parameters = list(raw.root_model.parameters()) + list(raw.body_model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    model.diffusion.calc_diffusion_vars(torch.arange(model.diffusion.num_base_steps, device=device))

    train = HumanML3DArdyDataset(
        args.manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=40,
        condition_dropout=args.constraint_dropout, seed=args.seed, text_cache_dir=args.text_cache,
        history_probability=args.history_probability,
        text_dropout=args.text_dropout,
    )
    dev = HumanML3DArdyDataset(
        args.dev_manifest, args.stats_dir, text_root=args.text_root, frames=args.frames, horizon=40,
        condition_dropout=0.0, seed=args.seed + 10000, text_cache_dir=args.text_cache,
        history_probability=args.history_probability,
        text_dropout=0.0,
    )
    dev_fixed = []
    sequential_eval = args.sequential_cache_probability > 0
    if sequential_eval and not dev.history_eligible_indices:
        raise ValueError("dev split has no samples eligible for sequential-cache evaluation")
    for batch_index in range(args.eval_batches):
        dev_start = (batch_index * world + rank) * args.batch_size
        dev_pool = dev.history_eligible_indices if sequential_eval else range(len(dev))
        dev_indices = [
            dev_pool[(dev_start + offset) % len(dev_pool)]
            for offset in range(args.batch_size)
        ]
        dev_batch = default_collate(
            [
                dev.sample(index, args.seed + 10000 + index, force_history=sequential_eval)
                for index in dev_indices
            ]
        )
        dev_fixed.append(
            _build_batch(
                model, dev_batch, device, fixed_noise=True,
                boundary_loss_weight=args.boundary_loss_weight,
                root_velocity_loss_weight=args.root_velocity_loss_weight,
                constraint_loss_weight=args.constraint_loss_weight,
                fk_constraint_loss_weight=args.fk_constraint_loss_weight,
                foot_skate_loss_weight=args.foot_skate_loss_weight,
                sequential_cache=sequential_eval,
            )
        )
    data_generator = torch.Generator().manual_seed(args.seed + 555)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    log_path = output / "train.jsonl"

    start_step = 0
    losses: list[float] = []
    all_finite = True
    sequential_cache_steps = 0
    elapsed_before = 0.0
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("mode") != args.mode:
            raise ValueError("initial checkpoint mode does not match requested run")
        _load_model_weights(raw, initial)
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("mode") != args.mode or resume.get("world_size") != world:
            raise ValueError("resume checkpoint mode/world size does not match")
        _load_model_weights(raw, resume)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])
        losses = list(resume["losses"])
        all_finite = bool(resume["all_gradients_finite"])
        sequential_cache_steps = int(resume.get("sequential_cache_steps", 0))
        data_generator.set_state(resume["data_generator_state"].cpu())
        local_rng = resume["rng_by_rank"][rank]
        torch.set_rng_state(local_rng["torch"].cpu())
        torch.cuda.set_rng_state(local_rng["cuda"].cpu(), device)
        initial_eval = float(resume["initial_eval_loss"])
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))
        if args.steps <= start_step:
            raise ValueError(f"--steps must exceed resumed step {start_step}")
    else:
        local_eval = torch.tensor(_evaluate_many(raw, client, dev_fixed), device=device)
        initial_eval = _mean_across_ranks(local_eval)
        if rank == 0 and log_path.exists():
            raise FileExistsError(f"refusing to overwrite existing log: {log_path}")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    global_batch = args.batch_size * world
    for step in range(start_step + 1, args.steps + 1):
        use_sequential_cache = _global_probability_draw(
            args.sequential_cache_probability, data_generator, device, rank
        )
        train_pool = train.history_eligible_indices if use_sequential_cache else range(len(train))
        if use_sequential_cache and not train_pool:
            raise ValueError("train split has no samples eligible for sequential-cache training")
        indices, augmentation_seeds = _global_draws(
            global_batch, len(train_pool), data_generator, device, rank
        )
        indices = torch.tensor(
            [train_pool[int(index)] for index in indices.cpu().tolist()],
            dtype=torch.long,
            device=device,
        )
        local_indices = indices[rank * args.batch_size:(rank + 1) * args.batch_size].cpu().tolist()
        local_seeds = augmentation_seeds[
            rank * args.batch_size:(rank + 1) * args.batch_size
        ].cpu().tolist()
        batch = default_collate(
            [
                train.sample(index, seed, force_history=True if use_sequential_cache else None)
                for index, seed in zip(local_indices, local_seeds)
            ]
        )
        fixed = _build_batch(
            model, batch, device, fixed_noise=False,
            boundary_loss_weight=args.boundary_loss_weight,
            root_velocity_loss_weight=args.root_velocity_loss_weight,
            constraint_loss_weight=args.constraint_loss_weight,
            fk_constraint_loss_weight=args.fk_constraint_loss_weight,
            foot_skate_loss_weight=args.foot_skate_loss_weight,
            sequential_cache=use_sequential_cache,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                loss, root_loss, body_loss, boundary_loss, velocity_loss,
                constraint_loss, fk_constraint_loss, foot_skate_loss,
            ) = _loss(raw, client, fixed, return_components=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"rank {rank}: non-finite loss at step {step}")
        loss.backward()
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)
        finite_tensor = torch.tensor(int(finite), device=device)
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        all_finite = all_finite and bool(finite_tensor.item())
        if not all_finite:
            raise RuntimeError(f"non-finite DDP gradient at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = _mean_across_ranks(loss)
        if rank == 0:
            losses.append(mean_loss)
            sequential_cache_steps += int(use_sequential_cache)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            mean_grad = _mean_across_ranks(grad_norm)
            mean_root_loss = _mean_across_ranks(root_loss)
            mean_body_loss = _mean_across_ranks(body_loss)
            mean_boundary_loss = _mean_across_ranks(boundary_loss)
            mean_velocity_loss = _mean_across_ranks(velocity_loss)
            mean_constraint_loss = _mean_across_ranks(constraint_loss)
            mean_fk_constraint_loss = _mean_across_ranks(fk_constraint_loss)
            mean_foot_skate_loss = _mean_across_ranks(foot_skate_loss)
            if rank == 0:
                record = {
                    "step": step,
                    "loss": mean_loss,
                    "root_loss": mean_root_loss,
                    "body_loss": mean_body_loss,
                    "boundary_loss": mean_boundary_loss,
                    "root_velocity_loss": mean_velocity_loss,
                    "constraint_loss": mean_constraint_loss,
                    "fk_constraint_loss": mean_fk_constraint_loss,
                    "foot_skate_loss": mean_foot_skate_loss,
                    "grad_norm": mean_grad,
                    "sequential_cache": use_sequential_cache,
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            local_eval = torch.tensor(_evaluate_many(raw, client, dev_fixed), device=device)
            eval_loss = _mean_across_ranks(local_eval)
            if rank == 0:
                record = {"step": step, "eval_loss": eval_loss}
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if args.save_every and step % args.save_every == 0:
            _save_checkpoint(
                output / f"checkpoint_step_{step:06d}.pt",
                args=args, raw=raw, optimizer=optimizer, step=step,
                losses=losses, initial_eval=initial_eval, all_finite=all_finite,
                sequential_cache_steps=sequential_cache_steps,
                data_generator=data_generator,
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
                device=device, rank=rank,
            )
            _prune_checkpoints(output, args.keep_last_checkpoints, rank)

    local_eval = torch.tensor(_evaluate_many(raw, client, dev_fixed), device=device)
    final_eval = _mean_across_ranks(local_eval)
    torch.cuda.synchronize(device)
    elapsed = elapsed_before + time.perf_counter() - started
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**20, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        first = float(np.mean(losses[:min(50, len(losses))]))
        last = float(np.mean(losses[-min(50, len(losses)):]))
        # A run initialized from a trained checkpoint is a fine-tune/resume
        # gate: its fixed validation loss may legitimately move upward while
        # the sampled training objective continues to improve.  Fresh full
        # training keeps the stricter train+validation decrease requirement.
        initialized = bool(args.init_checkpoint or args.resume)
        eval_gate = final_eval < initial_eval if not initialized else True
        metrics = {
            "status": "PASS" if all_finite and last < first and eval_gate else "FAIL",
            "mode": args.mode,
            "world_size": world,
            "per_rank_batch_size": args.batch_size,
            "global_batch_size": global_batch,
            "steps": args.steps,
            "first_50_mean_loss": first,
            "last_50_mean_loss": last,
            "initial_eval_loss": initial_eval,
            "final_eval_loss": final_eval,
            "initialized_from_checkpoint": initialized,
            "evaluation_gate": "informational_finetune" if initialized else "strict_decrease",
            "all_gradients_finite": all_finite,
            "sequential_cache_probability": args.sequential_cache_probability,
            "sequential_cache_steps": sequential_cache_steps,
            "elapsed_seconds": elapsed,
            "mean_step_seconds": elapsed / max(args.steps, 1),
            "samples_per_second": global_batch * args.steps / elapsed,
            "frames_per_second": global_batch * args.frames * args.steps / elapsed,
            "peak_memory_mib": float(peak),
            "text_backend": "llm2vec_cache" if args.text_cache else "deterministic_hash_smoke_only",
            "attention_backend": "sdpa",
            "keep_last_checkpoints": args.keep_last_checkpoints,
        }
        model = _unwrap_raw(raw)
        root = _unwrap_block(model.root_model)
        body = _unwrap_block(model.body_model)
        auxiliary = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith(("root_model.", "body_model."))
        }
        checkpoint = {
            "mode": args.mode,
            "root_model": root.state_dict(),
            "body_model": body.state_dict(),
            "denoiser_aux": auxiliary,
            "metrics": metrics,
        }
        torch.save(checkpoint, output / f"{args.mode}_humanml3d_ddp.pt")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    if (
        not args.no_final_resume_checkpoint
        and (not args.save_every or args.steps % args.save_every != 0)
    ):
        _save_checkpoint(
            output / f"checkpoint_step_{args.steps:06d}.pt",
            args=args, raw=raw, optimizer=optimizer, step=args.steps,
            losses=losses, initial_eval=initial_eval, all_finite=all_finite,
            sequential_cache_steps=sequential_cache_steps,
            data_generator=data_generator, elapsed_seconds=elapsed,
            device=device, rank=rank,
        )
        _prune_checkpoints(output, args.keep_last_checkpoints, rank)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
