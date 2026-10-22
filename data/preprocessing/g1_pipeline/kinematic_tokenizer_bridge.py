"""Project-normalized interface with the frozen released root-conditioning path."""
import torch
from motion_mamba2.models.latent_utils import HybridMotionConverter


class KinematicHybridConverter(HybridMotionConverter):
    def __init__(self, *, project_rep, released_converter, bridge):
        super().__init__(denoiser=released_converter.denoiser, autoencoder=bridge,
                         motion_rep=project_rep,
                         gen_horizon_len=released_converter.gen_horizon_len)
        self.released_converter = released_converter

    def get_hybrid_motion_from_explicit(self, motion, motion_len, motion_pad_mask):
        official = self.released_converter
        physical = self.motion_rep.unnormalize(motion)
        hybrid, mask = official.get_hybrid_motion_from_explicit(
            official.motion_rep.normalize(physical), motion_len, motion_pad_mask)
        # Preserve the caller's normalized global root without a redundant round trip.
        width = self.denoiser.nframe_root_dim
        root = self.motion_rep.extract_root(motion).reshape(motion.shape[0], -1, width)
        return torch.cat((root, hybrid[..., width:]), dim=-1), mask

    def get_explicit_motion_from_hybrid(self, hybrid_motion, motion_pad_mask,
                                      motion_len, motion_mask=None):
        official = self.released_converter
        root, latent = self.get_root_and_latent_body_motion_from_hybrid(hybrid_motion)
        physical_root = self.motion_rep.global_root_stats.unnormalize(root)
        released_root = official.motion_rep.global_root_stats.normalize(physical_root)
        released_hybrid = official.get_hybrid_motion_from_root_and_latent_body_motion(
            released_root, latent)
        decoded = official.get_explicit_motion_from_hybrid(
            released_hybrid, motion_pad_mask, motion_len, motion_mask)
        return self.motion_rep.normalize(official.motion_rep.unnormalize(decoded))
