"""
Hierarchical Noise Schedules for MDLM

Different masking rates per hierarchy level:
- Level 0 (Summary): Lower masking rate (less noisy)
- Level 1 (Content): Higher masking rate (more noisy)

This ensures summary tokens are denoised first, providing context for content generation.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class HierarchicalNoiseSchedule(nn.Module):
    """
    Hierarchical noise schedule where different hierarchy levels
    have different masking rates at each timestep.

    Key property: masking_rate[level_0] < masking_rate[level_1] for all t
    This means summary tokens (level 0) are always less masked than content (level 1).
    """

    def __init__(
        self,
        num_levels: int = 2,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        level_offsets: Optional[List[float]] = None,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.eps = eps

        # Level offsets control how much earlier each level is denoised
        # Higher offset = earlier denoising (less noisy at same timestep)
        if level_offsets is None:
            # Default: level 0 has offset 0.3 (30% ahead), level 1 has offset 0
            # This means at t=0.5, level 0 acts like t=0.2, level 1 acts like t=0.5
            level_offsets = [0.3 * (num_levels - 1 - k) / (num_levels - 1)
                           for k in range(num_levels)]

        self.register_buffer(
            'level_offsets',
            torch.tensor(level_offsets, dtype=torch.float32)
        )

        # Precompute schedules for each level
        schedules = []
        for k in range(num_levels):
            schedule = self._compute_schedule(level_offsets[k])
            schedules.append(schedule)

        # Shape: [num_levels, num_timesteps + 1]
        self.register_buffer(
            'masking_rates',
            torch.stack(schedules, dim=0)
        )

    def _compute_schedule(self, offset: float) -> torch.Tensor:
        """
        Compute masking rate schedule for a given level offset.

        Args:
            offset: How much to shift the schedule (0 = standard, >0 = less noisy)

        Returns:
            masking_rates: [num_timesteps + 1] tensor of masking probabilities
        """
        t = torch.linspace(0, 1, self.num_timesteps + 1)

        # Apply offset: shift t to make this level less noisy
        # t_effective = max(0, t - offset)
        t_effective = torch.clamp(t - offset, min=0.0)

        if self.schedule_type == "cosine":
            # Cosine schedule: mask_rate = 1 - cos(pi/2 * t_eff)^2
            cos_t = torch.cos(math.pi / 2 * t_effective)
            masking_rate = 1 - cos_t ** 2
        elif self.schedule_type == "linear":
            # Linear schedule
            masking_rate = t_effective
        elif self.schedule_type == "sqrt":
            # Square root schedule (slower start, faster end)
            masking_rate = torch.sqrt(t_effective)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Clamp to [eps, 1 - eps]
        masking_rate = torch.clamp(masking_rate, self.eps, 1 - self.eps)

        return masking_rate

    def get_masking_rate(
        self,
        t: torch.Tensor,
        hierarchy_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get masking rate for each token based on timestep and hierarchy level.

        Args:
            t: [B] or scalar - timestep indices (0 to num_timesteps)
            hierarchy_labels: [B, L] - hierarchy level for each token

        Returns:
            masking_rates: [B, L] - masking probability for each token
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        batch_size, seq_len = hierarchy_labels.shape

        # Get masking rates for all levels at timestep t
        # masking_rates shape: [num_levels, num_timesteps + 1]
        rates_at_t = self.masking_rates[:, t.long()]  # [num_levels, B]
        rates_at_t = rates_at_t.T  # [B, num_levels]

        # Gather rates based on hierarchy labels
        # hierarchy_labels: [B, L], rates_at_t: [B, num_levels]
        rates = torch.gather(
            rates_at_t.unsqueeze(2).expand(-1, -1, seq_len),
            dim=1,
            index=hierarchy_labels.unsqueeze(1)
        ).squeeze(1)  # [B, L]

        return rates

    def sample_masks(
        self,
        t: torch.Tensor,
        hierarchy_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample binary masks based on masking rates.

        Args:
            t: [B] - timestep indices
            hierarchy_labels: [B, L] - hierarchy level for each token

        Returns:
            masks: [B, L] - binary mask (1 = keep, 0 = mask)
        """
        masking_rates = self.get_masking_rate(t, hierarchy_labels)
        # Sample: mask token if random < masking_rate
        masks = (torch.rand_like(masking_rates) >= masking_rates).long()
        return masks

    def forward(
        self,
        t: torch.Tensor,
        hierarchy_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returns masking rates."""
        return self.get_masking_rate(t, hierarchy_labels)


class HierarchicalMasker(nn.Module):
    """
    Applies hierarchical masking to input sequences.
    """

    def __init__(
        self,
        noise_schedule: HierarchicalNoiseSchedule,
        mask_token_id: int,
        num_levels: int = 2,
    ):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.num_levels = num_levels
        # Level K is "unknown" for masked tokens
        self.unknown_level = num_levels

    def forward(
        self,
        input_ids: torch.Tensor,
        hierarchy_labels: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Apply hierarchical masking.

        Args:
            input_ids: [B, L] - original token IDs
            hierarchy_labels: [B, L] - ground truth hierarchy labels
            t: [B] - timestep for each sample
            attention_mask: [B, L] - attention mask (optional)

        Returns:
            dict with:
                - masked_ids: [B, L] - masked input IDs
                - mask: [B, L] - binary mask (1 = masked, 0 = kept)
                - hierarchy_input: [B, L] - hierarchy labels for model input
                                           (K for masked tokens, GT for unmasked)
        """
        # Sample masks based on hierarchical schedule
        keep_mask = self.noise_schedule.sample_masks(t, hierarchy_labels)
        mask = 1 - keep_mask  # 1 = masked, 0 = kept

        # Apply attention mask if provided (don't mask padding)
        if attention_mask is not None:
            mask = mask * attention_mask

        # Create masked input IDs
        masked_ids = input_ids.clone()
        masked_ids[mask.bool()] = self.mask_token_id

        # Create hierarchy input labels
        # Masked tokens get "unknown" level (K), unmasked keep ground truth
        hierarchy_input = hierarchy_labels.clone()
        hierarchy_input[mask.bool()] = self.unknown_level

        return {
            'masked_ids': masked_ids,
            'mask': mask,
            'hierarchy_input': hierarchy_input,
            'keep_mask': keep_mask,
        }


if __name__ == "__main__":
    # Test the hierarchical noise schedule
    schedule = HierarchicalNoiseSchedule(
        num_levels=2,
        num_timesteps=1000,
        schedule_type="cosine",
    )

    print("Level offsets:", schedule.level_offsets)
    print("Masking rates shape:", schedule.masking_rates.shape)

    # Compare masking rates at different timesteps
    for t in [100, 500, 900]:
        rate_0 = schedule.masking_rates[0, t].item()
        rate_1 = schedule.masking_rates[1, t].item()
        print(f"t={t}: Level 0 rate = {rate_0:.3f}, Level 1 rate = {rate_1:.3f}")
        assert rate_0 <= rate_1, "Level 0 should be less masked than Level 1"

    print("\nAll assertions passed! Level 0 is always less masked than Level 1.")
