import json

import numpy as np
import torch

from ardy_data.humanml3d import (
    CORE_PARENTS,
    HumanML3DArdyDataset,
    SMPL22_TO_CORE,
    _axis_angle_to_matrix,
    _global_from_local,
    _local_from_global,
    sample_control_condition,
)


def test_dataset_stochastic_view_is_seeded_and_direct_index_is_fixed(tmp_path):
    features = np.arange(120 * 330, dtype=np.float32).reshape(120, 330)
    feature_path = tmp_path / "sample.npy"
    np.save(feature_path, features)
    np.save(tmp_path / "mean.npy", np.zeros(334, dtype=np.float32))
    np.save(tmp_path / "std.npy", np.ones(334, dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "sample", "feature_path": str(feature_path)}) + "\n")
    dataset = HumanML3DArdyDataset(manifest, tmp_path, frames=80, seed=17)

    fixed_a = dataset[0]
    fixed_b = dataset[0]
    seeded_a = dataset.sample(0, 1234)
    seeded_b = dataset.sample(0, 1234)
    seeded_c = dataset.sample(0, 5678)

    torch.testing.assert_close(fixed_a["motion"], fixed_b["motion"])
    torch.testing.assert_close(seeded_a["motion"], seeded_b["motion"])
    assert not torch.equal(seeded_a["motion"], seeded_c["motion"])


def test_dataset_can_force_uniform_second_window_history(tmp_path):
    features = np.zeros((120, 330), dtype=np.float32)
    features[:, 0] = np.arange(120)
    feature_path = tmp_path / "sample.npy"
    np.save(feature_path, features)
    np.save(tmp_path / "mean.npy", np.zeros(334, dtype=np.float32))
    np.save(tmp_path / "std.npy", np.ones(334, dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "sample", "feature_path": str(feature_path), "frames": 120}) + "\n"
    )
    dataset = HumanML3DArdyDataset(manifest, tmp_path, frames=80, seed=17)

    first = dataset.sample(0, 1234, force_history=False)
    second = dataset.sample(0, 1234, force_history=True)

    assert first["history_frames"] == 0
    assert second["history_frames"] == 40
    assert dataset.history_eligible_indices == [0]
    assert torch.isclose(second["motion"][39, 0], torch.tensor(0.0))


def test_retarget_map_is_explicit_and_core_indices_are_unique():
    assert len(SMPL22_TO_CORE) == 22
    assert set(SMPL22_TO_CORE) == set(range(22))
    assert len(set(SMPL22_TO_CORE.values())) == 22
    assert set(SMPL22_TO_CORE.values()) <= set(range(27))


def test_local_global_rotation_roundtrip():
    generator = torch.Generator().manual_seed(7)
    axis_angle = torch.randn((6, 27, 3), generator=generator) * 0.2
    local = _axis_angle_to_matrix(axis_angle)
    global_rots = _global_from_local(local, CORE_PARENTS)
    recovered = _local_from_global(global_rots, CORE_PARENTS)
    torch.testing.assert_close(recovered, local, atol=2e-6, rtol=2e-6)


def test_control_sampler_in_and_out_of_horizon_masks_gt_values():
    features = torch.arange(100 * 330, dtype=torch.float32).reshape(100, 330)
    for control in ("root_trajectory", "root_waypoint", "full_keyframe", "joint_position", "joint_rotation"):
        result = sample_control_condition(
            features,
            skeleton=None,
            rng=np.random.default_rng(11),
            horizon=40,
            control=control,
            out_of_horizon=True,
        )
        mask = result["motion_mask"]
        assert mask.any()
        assert result["frame"] > 40
        torch.testing.assert_close(result["observed_motion"][mask], features[mask])
        assert torch.count_nonzero(result["observed_motion"][~mask]) == 0


def test_condition_dropout_clears_motion_condition():
    features = torch.randn(50, 330)
    result = sample_control_condition(
        features,
        skeleton=None,
        rng=np.random.default_rng(1),
        condition_dropout=1.0,
    )
    assert result["condition_dropped"] is True
    assert not result["motion_mask"].any()
    assert not result["observed_motion"].any()


def test_cached_caption_and_text_dropout_match_embedding(tmp_path):
    features = np.zeros((80, 330), dtype=np.float32)
    feature_path = tmp_path / "sample.npy"
    np.save(feature_path, features)
    np.save(tmp_path / "mean.npy", np.zeros(334, dtype=np.float32))
    np.save(tmp_path / "std.npy", np.ones(334, dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "sample", "feature_path": str(feature_path), "frames": 80}) + "\n"
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "embeddings.npy", np.ones((1, 4096), dtype=np.float16))
    (cache / "index.json").write_text(json.dumps({
        "samples": [{"id": "sample", "row": 0, "caption": "cached caption"}]
    }))

    kept = HumanML3DArdyDataset(
        manifest, tmp_path, text_cache_dir=cache, text_dropout=0.0
    )[0]
    dropped = HumanML3DArdyDataset(
        manifest, tmp_path, text_cache_dir=cache, text_dropout=1.0
    )[0]

    assert kept["text"] == "cached caption"
    assert not kept["text_dropped"]
    assert torch.count_nonzero(kept["text_feature"]) == 4096
    assert dropped["text"] == ""
    assert dropped["text_dropped"]
    assert torch.count_nonzero(dropped["text_feature"]) == 0


def test_short_sequence_cannot_claim_out_of_horizon():
    result = sample_control_condition(
        torch.randn(20, 330),
        skeleton=None,
        rng=np.random.default_rng(4),
        horizon=40,
        out_of_horizon=True,
    )
    assert result["out_of_horizon"] is False


def test_history_window_places_constraints_relative_to_generation():
    features = torch.randn(120, 330)
    inside = sample_control_condition(
        features,
        skeleton=None,
        rng=np.random.default_rng(8),
        horizon=40,
        generation_start=40,
        out_of_horizon=False,
    )
    future = sample_control_condition(
        features,
        skeleton=None,
        rng=np.random.default_rng(9),
        horizon=40,
        generation_start=40,
        out_of_horizon=True,
    )
    assert 40 <= inside["frame"] < 80
    assert future["frame"] > 80
