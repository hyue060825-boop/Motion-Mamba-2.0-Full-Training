"""Precompute deterministic HumanML3D caption embeddings with local LLM2Vec."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))

from ardy_data import select_cached_caption  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--text-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    records = []
    seen = set()
    for manifest in args.manifest:
        for line in Path(manifest).read_text().splitlines():
            record = json.loads(line)
            if record["id"] not in seen:
                records.append(record)
                seen.add(record["id"])
    if args.max_samples is not None:
        records = records[:args.max_samples]
    captions = [select_cached_caption(record["id"], args.text_root, args.seed) for record in records]
    if not all(captions):
        raise ValueError("all cached samples must have a non-empty caption")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embedding_path = output / "embeddings.npy"
    progress_path = output / "progress.json"
    if embedding_path.exists():
        embeddings = np.load(embedding_path, mmap_mode="r+")
        if embeddings.shape != (len(records), 4096):
            raise ValueError(f"existing cache shape {embeddings.shape} does not match {(len(records), 4096)}")
        start = json.loads(progress_path.read_text()).get("completed", 0) if progress_path.exists() else 0
    else:
        embeddings = np.lib.format.open_memmap(
            embedding_path, mode="w+", dtype=np.float16, shape=(len(records), 4096)
        )
        start = 0

    from ardy.model.load_model import load_text_encoder

    encoder = load_text_encoder(mode="local", device=args.device)
    started = time.perf_counter()
    for index in range(start, len(records)):
        with torch.inference_mode():
            feature, lengths = encoder([captions[index]])
        if lengths != [1] or feature.shape != (1, 1, 4096) or not torch.isfinite(feature).all():
            raise RuntimeError(f"invalid text embedding at row {index}")
        embeddings[index] = feature[0, 0].float().cpu().numpy().astype(np.float16)
        if (index + 1) % args.flush_every == 0 or index + 1 == len(records):
            embeddings.flush()
            progress_path.write_text(json.dumps({"completed": index + 1, "total": len(records)}) + "\n")
            elapsed = time.perf_counter() - started
            rate = (index + 1 - start) / max(elapsed, 1e-9)
            print(json.dumps({"completed": index + 1, "total": len(records), "samples_per_second": rate}), flush=True)

    samples = [
        {"id": record["id"], "row": index, "caption": captions[index]}
        for index, record in enumerate(records)
    ]
    (output / "index.json").write_text(
        json.dumps({"format_version": 1, "dtype": "float16", "dim": 4096, "samples": samples}) + "\n"
    )
    (output / "complete.json").write_text(
        json.dumps({"status": "PASS", "samples": len(records), "device": args.device}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
