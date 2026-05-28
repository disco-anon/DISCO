"""
Model-agnostic Self-Paced Learning controller.

Ported verbatim (same formulas, same default values, same edge cases) from the
SPL implementation embedded in ``ts_benchmark/baselines/catch/CATCH.py``:

    - state fields:       CATCH.py:96-103
    - _spl_compute_loss:  CATCH.py:160-251
    - cooldown logic:     CATCH.py:197-206
    - blowup fuse:        CATCH.py:399-425

The goal is a model-agnostic utility so any reconstruction-style baseline can
plug SPL in by passing a per-sample loss tensor to ``compute_loss`` and driving
the lifecycle hooks. When paired with CATCH's own SPL, the dynamics should
match one-to-one (enabling fair cross-baseline comparison).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SPLConfig:
    """SPL hyperparameters — defaults mirror CATCH's ``TransformerConfig``."""

    enable_spl: bool = True
    spl_start_epoch: int = 1              # first SPL-active epoch (0-indexed)
    spl_init_weight: float = 0.5          # starting target quantile for lambda
    spl_target_quantile: float = 0.95     # ending target quantile for lambda
    spl_gamma: float = 0.9                # EMA coefficient on lambda
    spl_temperature: float = 1.0          # sigmoid temperature (scaled by buffer.std())
    spl_min_weight: float = 0.3           # weight floor (applied to the down-weighted side)
    spl_blowup_ratio: float = 2.0         # fuse trigger: val_loss > ratio * warmup_val
    spl_cooldown_epochs: int = 1          # epochs at the tail with SPL disabled
    spl_buffer_size: int = 2048           # target sample count in rolling buffer
    # Curriculum direction:
    #   "easy_first" (default, matches CATCH SPL): weight high for difficulty < λ
    #     → focuses gradient on samples the model already fits well.
    #   "hard_first" (Hard Example Mining variant): weight high for difficulty > λ
    #     → focuses gradient on samples the model struggles with. Useful on
    #     datasets with sparse signals (MSL, SWAT) where SensitiveHUE's
    #     α-weighting already targets uncertain samples and easy_first SPL
    #     would cancel out that focus.
    spl_mode: str = "easy_first"
    # Difficulty signal source. The caller decides how to compute the
    # per-sample difficulty tensor it passes to ``compute_loss``; this flag is
    # purely a passthrough config field that the caller can read to pick a
    # policy. See SensitiveHUE_disco for the canonical interpretation:
    #   "mse" (default, matches CATCH's MSE-as-difficulty): per-sample MSE.
    #     Independent of the α-weighted NLL gradient signal — gives SPL an
    #     orthogonal handle that works well when the loss curvature is mild
    #     (SMAP/SMD/PSM).
    #   "nll" : per-sample α-weighted NLL (the actual training loss, batch-
    #     shifted to ≥ 0). Same source as the gradient → SPL ranking aligned
    #     with what α-weighting already optimizes. Reduces signal mismatch
    #     on datasets where MSE-based SPL fights α-weighting (MSL/SWAT).
    spl_difficulty_source: str = "mse"


class SPLController:
    """Stateful SPL controller. One instance per training run.

    Lifecycle (caller responsibilities):
        spl = SPLController(cfg, num_epochs=N)
        for epoch in range(N):                                    # 0-indexed
            phase = spl.on_epoch_start(epoch)
            for batch in loader:
                per_sample = compute_per_sample_loss(...)         # (B,)
                loss, avg_w = spl.compute_loss(per_sample)
                loss.backward(); ...
            val_loss = validate(...)
            if epoch == cfg.spl_start_epoch - 1:
                spl.on_warmup_end(val_loss, model)
            fuse = spl.on_epoch_end(val_loss, model)
            if fuse["fused"]:
                # caller should reset its EarlyStopping (seeded with spl.warmup_vali)
                ...

    Key invariant: if ``enable_spl=False`` or SPL is disabled/warmup/cooldown,
    ``compute_loss`` returns ``per_sample.mean()`` so subclasses can wire it in
    unconditionally and get vanilla behavior when SPL is off.
    """

    def __init__(self, config: SPLConfig, num_epochs: int):
        self.config = config
        self.num_epochs = num_epochs
        self.current_epoch: int = 0
        self.threshold: Optional[torch.Tensor] = None   # scalar tensor, EMA
        self.loss_buffer: list = []                     # list of detached (B,) tensors
        self.disabled: bool = False
        self.warmup_vali: Optional[float] = None
        self.warmup_state: Optional[dict] = None

    # ------------------------------------------------------------------ phase

    def _phase(self) -> str:
        if not self.config.enable_spl or self.disabled:
            return "disabled"
        if self.current_epoch < self.config.spl_start_epoch:
            return "warmup"
        # auto-disable cooldown when SPL budget is too short (matches CATCH:197-200)
        effective_cd = self.config.spl_cooldown_epochs
        spl_budget = self.num_epochs - self.config.spl_start_epoch
        if spl_budget <= 2:
            effective_cd = 0
        cooldown_start = self.num_epochs - effective_cd
        if self.current_epoch >= cooldown_start:
            return "cooldown"
        return "active"

    def on_epoch_start(self, epoch_idx: int) -> str:
        """Set current epoch (0-indexed) and return the phase string."""
        self.current_epoch = epoch_idx
        return self._phase()

    # ---------------------------------------------------------------- loss op

    def compute_loss(
        self,
        loss: torch.Tensor,
        difficulty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Core SPL loss reduction.

        Args:
            loss: shape (B,), per-sample loss that gradients flow through. May
                have any sign (e.g., NLL can be negative).
            difficulty: optional shape (B,) per-sample "hardness" signal that
                drives the SPL machinery (buffer, lambda quantile, sigmoid
                weighting). Must be non-negative (the Winsorize clip uses
                ``99th-percentile × 10`` as an upper bound, which is only
                meaningful on a non-negative scale). Defaults to ``loss``
                (matches CATCH's MSE-only case where loss == difficulty).

        Returns:
            (scalar_loss, avg_weight). When SPL is not active, returns
            (loss.mean(), 1.0).
        """
        if difficulty is None:
            difficulty = loss
        # Difficulty drives the buffer/lambda/weight; it must be detached.
        difficulty_detached = difficulty.detach()

        # 1) Winsorize extremely hard samples against the current buffer,
        #    then push the winsorized difficulty. Only kicks in after at least
        #    2 batches so early statistics aren't over-clipped.
        if len(self.loss_buffer) >= 2:
            buf_cat = torch.cat(self.loss_buffer)
            clip_hi = torch.quantile(buf_cat, 0.99).detach() * 10.0
            difficulty_detached = torch.clamp(difficulty_detached, max=clip_hi)
        self.loss_buffer.append(difficulty_detached)

        # trim buffer to ~spl_buffer_size samples worth (using current batch size)
        max_batches = max(1, self.config.spl_buffer_size // loss.shape[0])
        while len(self.loss_buffer) > max_batches:
            self.loss_buffer.pop(0)

        # 2) If SPL not active in this phase, behave as vanilla mean reduction.
        if self._phase() != "active":
            return loss.mean(), 1.0

        buffer = torch.cat(self.loss_buffer)

        # 3) First activation: initialize threshold at the init quantile.
        if self.threshold is None:
            self.threshold = torch.quantile(
                buffer, self.config.spl_init_weight
            ).detach()

        # 4) Ramp the target quantile linearly over active epochs, then EMA.
        active_epochs = max(1, self.num_epochs - self.config.spl_start_epoch)
        progress = (self.current_epoch - self.config.spl_start_epoch) / active_epochs
        progress = min(max(progress, 0.0), 1.0)
        target_q = (
            self.config.spl_init_weight
            + (self.config.spl_target_quantile - self.config.spl_init_weight)
            * progress
        )
        target_q = min(max(target_q, 0.05), 0.99)
        target_thr = torch.quantile(buffer, target_q).detach()
        self.threshold = (
            self.config.spl_gamma * self.threshold
            + (1.0 - self.config.spl_gamma) * target_thr
        ).detach()

        # 5) Adaptive temperature (scaled by buffer.std) → sigmoid → weight floor.
        loss_std = buffer.std().detach() + 1e-8
        scaled_temp = self.config.spl_temperature * loss_std
        if self.config.spl_mode == "hard_first":
            # high weight on samples ABOVE λ (Hard Example Mining)
            sig_arg = (difficulty.detach() - self.threshold) / scaled_temp
        else:  # "easy_first" — original CATCH behavior
            # high weight on samples BELOW λ (Curriculum Learning)
            sig_arg = (self.threshold - difficulty.detach()) / scaled_temp
        raw_weight = torch.sigmoid(sig_arg)
        min_w = self.config.spl_min_weight
        weight = min_w + (1.0 - min_w) * raw_weight

        # 6) Normalized weighted sum of the actual backward loss.
        weighted = (loss * weight).sum() / (weight.sum() + 1e-8)
        return weighted, weight.mean().item()

    # ---------------------------------------------------------------- fuse path

    def on_warmup_end(self, val_loss: float, model: torch.nn.Module) -> None:
        """Save the warmup-ended validation loss and model weights as the fuse
        baseline. Call this at ``epoch == spl_start_epoch - 1``."""
        if not self.config.enable_spl or self.disabled:
            return
        self.warmup_vali = float(val_loss)
        self.warmup_state = copy.deepcopy(model.state_dict())

    def on_epoch_end(self, val_loss: float, model: torch.nn.Module) -> dict:
        """Blowup check. If triggered, roll back model to warmup weights and
        permanently disable SPL. Returns ``{'fused': bool, 'reason': str|None}``.

        The caller is responsible for resetting any downstream state (e.g.,
        EarlyStopping) when ``fused=True``.
        """
        result = {"fused": False, "reason": None}
        if (
            not self.config.enable_spl
            or self.disabled
            or self.current_epoch < self.config.spl_start_epoch
            or self.warmup_vali is None
        ):
            return result

        # Lower validation loss is better.  The original CATCH fuse used
        # ``val > ratio * warmup`` because CATCH's validation loss is positive.
        # Probabilistic models such as OmniAnomaly can have negative NLL/ELBO
        # values, where that formula incorrectly treats a better value (more
        # negative) as a blow-up.  This threshold preserves the positive-loss
        # behavior and also works for negative baselines.
        fuse_threshold = self.warmup_vali + (
            self.config.spl_blowup_ratio - 1.0
        ) * abs(self.warmup_vali)
        bad = (not math.isfinite(val_loss)) or (val_loss > fuse_threshold)
        if bad:
            model.load_state_dict(self.warmup_state)
            self.disabled = True
            self.threshold = None
            self.loss_buffer = []
            result = {
                "fused": True,
                "reason": (
                    f"val={val_loss:.4f} > fuse_threshold={fuse_threshold:.4f} "
                    f"(warmup={self.warmup_vali:.4f}, "
                    f"ratio={self.config.spl_blowup_ratio:g})"
                ),
            }
        return result

    # ---------------------------------------------------------------- helpers

    @property
    def threshold_scalar(self) -> float:
        """Current lambda as a Python float (0.0 before first activation)."""
        return self.threshold.item() if self.threshold is not None else 0.0
