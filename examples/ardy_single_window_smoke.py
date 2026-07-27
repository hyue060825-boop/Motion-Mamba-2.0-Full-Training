"""Milestone B, final checklist items: swap `ArdyMambaBlock` into a REAL,
loaded ARDY denoiser and run one actual generation call (one autoregressive
window, several denoising steps), reusing `scripts/generate.py`'s own
loading/invocation pattern so this isn't a hand-rolled approximation of
ARDY's real call path.

Randomly-initialized Mamba weights -> expect GARBAGE motion. That's fine and
expected: this checks structural correctness (does it run, are shapes
right, is output finite), not motion quality, which needs actual training.

Run with (on the server):

CUDA_VISIBLE_DEVICES=5 \
MAMBA_TRITON_OVERLAY=/data3/zts/packages/mamba_op_triton \
MAMBA_REPO=/data3/zts/packages/state-spaces-mamba \
PYTHONPATH=/data3/zts/ardy \
/data3/zts/conda_envs/ardy_env/bin/python \
    examples/ardy_single_window_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get("MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 2
    torch.empty(1, device="cuda")

    sys.path.insert(0, TRITON_OVERLAY)
    sys.path.insert(0, MAMBA_REPO)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from ardy.model import DEFAULT_MODEL, load_model
    from ardy.model.loading import get_env_var
    from ardy.model.registry import resolve_model_name
    from ardy.motion_rep.tools import length_to_mask
    from ardy.tools import seed_everything

    from ardy_adapter import ArdyMambaBlock

    device = "cuda:0"
    checkpoints_dir = get_env_var("CHECKPOINTS_DIR")
    resolved_model = resolve_model_name(DEFAULT_MODEL, checkpoints_dir=checkpoints_dir)
    print(f"Loading {resolved_model} ...")
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)
    print("Loaded.")

    denoiser = model.denoiser
    root_kwargs = dict(
        input_dim=-1,
        add_input_proj=False,
        output_dim=denoiser.root_model.output_linear.out_features,
        skeleton=denoiser.motion_rep.skeleton,
        llm_shape=denoiser.llm_shape,
        use_text_mask=denoiser.root_model.use_text_mask,
        latent_dim=denoiser.latent_dim,
        num_layers=8,
        input_first_heading_angle=hasattr(denoiser.root_model, "linear_first_heading_angle"),
        positional_encoding_mode=denoiser.positional_encoding_mode,
        device=device,
    )
    body_kwargs = dict(root_kwargs, output_dim=denoiser.body_model.output_linear.out_features)

    print("Constructing ArdyMambaBlock replacements (random init -- expect garbage motion, structural test only)...")
    mamba_root = ArdyMambaBlock(**root_kwargs).to(device)
    mamba_body = ArdyMambaBlock(**body_kwargs).to(device)

    original_root, original_body = denoiser.root_model, denoiser.body_model
    denoiser.root_model = mamba_root
    denoiser.body_model = mamba_body
    print("Swapped denoiser.root_model / denoiser.body_model -> ArdyMambaBlock.")

    fps = model.motion_rep.fps
    gen_horizon_len = model.gen_horizon_len
    num_frames = gen_horizon_len  # exactly one autoregressive window
    text = "A person walks forward."
    num_samples = 1

    seed_everything(0)
    texts = [text] * num_samples
    lengths = torch.tensor([num_frames] * num_samples, device=device)
    pad_mask = length_to_mask(lengths)
    first_heading_angle = torch.zeros(num_samples, device=device)

    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = min(2, num_base_steps)  # minimal, this is a structural smoke test

    print(
        f"Generating {num_frames} frames ({num_frames / fps:.2f}s at {fps}fps), "
        f"{diffusion_steps} denoising steps, ArdyMambaBlock active..."
    )
    try:
        with torch.no_grad():
            motion = model(
                texts,
                num_frames,
                num_denoising_steps=diffusion_steps,
                pad_mask=pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=(2.0, 2.0),
                crop_history_length=None,
            )
        print("SINGLE_WINDOW_SMOKE_OK")
        print(f"motion output type={type(motion)}")
        if hasattr(motion, "shape"):
            print(f"motion shape={tuple(motion.shape)} dtype={motion.dtype} finite={torch.isfinite(motion).all().item()}")
        elif isinstance(motion, dict):
            for k, v in motion.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={tuple(v.shape)} finite={torch.isfinite(v).all().item() if v.is_floating_point() else 'n/a'}")
    except Exception as e:
        print(f"SINGLE_WINDOW_SMOKE_FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        denoiser.root_model, denoiser.body_model = original_root, original_body

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
