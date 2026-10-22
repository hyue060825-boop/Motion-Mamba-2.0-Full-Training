# Third-party notices

- **NVIDIA ARDY:** underlying G1 motion/codec/skeleton interfaces and associated derived implementation. Copyright (c)2026 NVIDIA CORPORATION & AFFILIATES. Source license: Apache-2.0. The original SPDX headers are retained; `third_party/LICENSE_NVIDIA_APACHE_2_0.txt` contains the supplied license. Public import names were reorganized, not represented as independent authorship. Upstream pretrained weights have their own model terms.
- **state-spaces/mamba / Dao AI Lab / Goombalab:** Mamba/Mamba-3 implementation. Exact server source snapshot, LICENSE, AUTHORS and SHA manifest are under `third_party/mamba`. Git metadata was unavailable; no commit was fabricated. The native stateful chunk integration is in `motion_mamba2.kernels`.
- **Triton, PyTorch and other Python dependencies:** installed externally under their respective licenses; binary environments/caches are not republished here.
- **GMR and other retargeting dependencies:** external prerequisites referenced by the data-processing modules. Obtain their pinned upstream sources under their respective licenses; downloaded dependency trees are excluded from the public source bundle.
- **LLM2Vec / Meta-Llama / Hugging Face model assets:** external text-model and pretrained assets, not included weights. Obtain access and follow their original terms.
- **BONES-SEED, HiPHI, HumanML3D and source motion collections:** original datasets and bulk annotations are excluded. The publication of processing code is not a grant of dataset redistribution rights.

Historical names retained in attribution, upstream asset IDs and checkpoint-format compatibility do not name the public project. Other published motion-model baselines remain distinct methods and are not renamed as Motion Mamba 2.0.

- **External comparison methods (DART, DiP, FloodDiffusion2, OMG, CMDM, MotionStreamer, MotionMamba):** our G1 adapters and admission/training scripts are separate from the upstream repositories. `COMPARISON_SOURCES.json` and `COMPARISON_SOURCE_HASHES.json` record available source identities and licenses; upstream source/weights are not silently relicensed or renamed as our model.
