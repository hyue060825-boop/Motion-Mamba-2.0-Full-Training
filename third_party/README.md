# Sequence backend and upstream components

`mamba/` is the exact source archive used by the executed runtime, including Mamba-3. Its Apache-2.0 LICENSE/AUTHORS and per-source SHA-256 are preserved. No upstream Git revision is asserted because the server archive had no Git metadata.

In a new isolated environment with the recorded PyTorch/CUDA toolkit, build the CUDA extension from that snapshot rather than downloading an unrelated version:

```bash
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps ./third_party/mamba
python -m pip install --target "$MM2_WORKSPACE/deps/triton" --no-deps triton==3.5.1
export MM2_TRITON_PATH="$MM2_WORKSPACE/deps/triton"
export MM2_MAMBA_SOURCE="$PWD/third_party/mamba"
```

The exact package versions and CUDA compiler/driver must be compatible. The release validation uses the existing environment without installing or replacing its global packages. Rebuilding a CUDA extension on another platform is a separate prerequisite, not a claimed completed training run.

The core motion/codec interfaces derive from NVIDIA's Apache-2.0 ARDY implementation. Its pretrained G1 codec/frontends remain upstream assets and must be obtained under their own published terms. Their original Hugging Face identifiers remain in the pretrained asset registry and legacy metadata adapter; these are not the Motion Mamba 2.0 project name.
