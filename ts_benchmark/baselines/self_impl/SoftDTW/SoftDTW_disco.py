import numpy as np
import pandas as pd

from ts_benchmark.baselines.self_impl.SoftDTW.SoftDTW import (
    SoftDTW,
    _soft_dtw_distance,
)


DEFAULT_SPL_PARAMS = {
    "num_epochs": 5,
    "num_prototypes": 8,
    "prototype_quantile": 0.75,
    "score_log_every": 500,
    "enable_spl": True,
    "spl_start_epoch": 0,
    "spl_init_weight": 0.5,
    "spl_target_quantile": 0.95,
    "spl_gamma": 0.9,
    "spl_temperature": 1.0,
    "spl_min_weight": 0.3,
    "spl_cooldown_epochs": 0,
    "spl_buffer_size": 2048,
    "spl_mode": "easy_first",
}


class SoftDTW_disco(SoftDTW):
    """
    SoftDTW + disco variant.

    This keeps the same CATCH-framework interface as SoftDTW. The change is in
    the normal reference set: instead of one global mean prototype, it uses SPL
    weights to choose multiple representative normal windows. At inference, a
    window score is the minimum SoftDTW distance to this prototype set.
    """

    def __init__(self, **kwargs):
        merged = dict(DEFAULT_SPL_PARAMS)
        merged.update(kwargs)
        super().__init__(**merged)
        self.model_name = "SoftDTW_disco"
        self.spl_threshold_ = None
        self.spl_loss_buffer_ = []
        self.prototypes_ = None

    def _spl_phase(self, epoch):
        if not self.config.enable_spl:
            return "disabled"
        if epoch < self.config.spl_start_epoch:
            return "warmup"
        effective_cd = self.config.spl_cooldown_epochs
        spl_budget = self.config.num_epochs - self.config.spl_start_epoch
        if spl_budget <= 2:
            effective_cd = 0
        cooldown_start = self.config.num_epochs - effective_cd
        if epoch >= cooldown_start:
            return "cooldown"
        return "active"

    def _spl_weights(self, distances, epoch):
        difficulty = np.asarray(distances, dtype=np.float64)
        difficulty = difficulty - np.min(difficulty)

        difficulty_for_buffer = difficulty.copy()
        if len(self.spl_loss_buffer_) >= 2:
            buffer = np.concatenate(self.spl_loss_buffer_)
            clip_hi = np.quantile(buffer, 0.99) * 10.0
            difficulty_for_buffer = np.minimum(difficulty_for_buffer, clip_hi)

        self.spl_loss_buffer_.append(difficulty_for_buffer)
        max_batches = max(1, int(self.config.spl_buffer_size) // len(difficulty))
        while len(self.spl_loss_buffer_) > max_batches:
            self.spl_loss_buffer_.pop(0)

        if self._spl_phase(epoch) != "active":
            return np.ones_like(difficulty, dtype=np.float64), 1.0, 0.0

        buffer = np.concatenate(self.spl_loss_buffer_)
        if self.spl_threshold_ is None:
            self.spl_threshold_ = np.quantile(buffer, self.config.spl_init_weight)

        active_epochs = max(1, self.config.num_epochs - self.config.spl_start_epoch)
        progress = (epoch - self.config.spl_start_epoch) / active_epochs
        progress = min(max(progress, 0.0), 1.0)
        target_q = (
            self.config.spl_init_weight
            + (self.config.spl_target_quantile - self.config.spl_init_weight)
            * progress
        )
        target_q = min(max(target_q, 0.05), 0.99)
        target_thr = np.quantile(buffer, target_q)
        self.spl_threshold_ = (
            self.config.spl_gamma * self.spl_threshold_
            + (1.0 - self.config.spl_gamma) * target_thr
        )

        scaled_temp = self.config.spl_temperature * (np.std(buffer) + 1e-8)
        if self.config.spl_mode == "hard_first":
            sig_arg = (difficulty - self.spl_threshold_) / scaled_temp
        else:
            sig_arg = (self.spl_threshold_ - difficulty) / scaled_temp
        raw_weight = 1.0 / (1.0 + np.exp(-sig_arg))
        min_w = self.config.spl_min_weight
        weight = min_w + (1.0 - min_w) * raw_weight
        return weight, float(np.mean(weight)), float(self.spl_threshold_)

    def _select_prototypes(self, windows, distances, weight):
        n_prototypes = max(1, int(self.config.num_prototypes))
        if len(windows) <= n_prototypes:
            return windows.copy()

        score = np.asarray(weight, dtype=np.float64)
        if np.allclose(score.max(), score.min()):
            score = -np.asarray(distances, dtype=np.float64)

        keep_cutoff = np.quantile(score, 1.0 - self.config.prototype_quantile)
        candidate_idx = np.flatnonzero(score >= keep_cutoff)
        if len(candidate_idx) < n_prototypes:
            candidate_idx = np.argsort(score)[-n_prototypes:]

        ordered = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
        pick_pos = np.linspace(0, len(ordered) - 1, n_prototypes, dtype=np.int64)
        selected = ordered[pick_pos]
        return windows[selected].copy()

    def _fit_prototype(self, values):
        windows = self._make_windows(
            values,
            stride=self.config.train_stride,
            max_windows=self.config.max_train_windows,
        )
        if windows.size == 0:
            raise ValueError(
                f"Training data length must be at least seq_len={self.config.seq_len}."
            )

        prototype = windows.mean(axis=0)
        distances = np.array(
            [_soft_dtw_distance(window, prototype, self.config.gamma) for window in windows],
            dtype=np.float64,
        )
        weight = np.ones(len(windows), dtype=np.float64)
        if not self.config.enable_spl:
            self.prototypes_ = self._select_prototypes(windows, distances, weight)
            return prototype

        for epoch in range(int(self.config.num_epochs)):
            distances = np.array(
                [
                    _soft_dtw_distance(window, prototype, self.config.gamma)
                    for window in windows
                ],
                dtype=np.float64,
            )
            weight, avg_weight, threshold = self._spl_weights(distances, epoch)
            prototype = np.average(windows, axis=0, weights=weight)
            print(
                f"[SoftDTW_disco] Epoch {epoch + 1:02d}/{self.config.num_epochs} "
                f"[{self._spl_phase(epoch)}] dist={np.mean(distances):.6f} "
                f"avg_w={avg_weight:.4f} lambda={threshold:.6f}"
            )
        self.prototypes_ = self._select_prototypes(windows, distances, weight)
        print(f"[SoftDTW_disco] selected prototypes: {len(self.prototypes_)}")
        return prototype

    def _score_windows(self, values, stride):
        starts = self._window_starts(values.shape[0], stride)
        scores = np.empty(len(starts), dtype=np.float64)
        prototypes = self.prototypes_
        if prototypes is None or len(prototypes) == 0:
            prototypes = np.expand_dims(self.prototype_, axis=0)

        total = len(starts)
        log_every = int(getattr(self.config, "score_log_every", 0) or 0)
        for idx, start in enumerate(starts):
            window = values[start:start + self.config.seq_len]
            scores[idx] = min(
                _soft_dtw_distance(window, prototype, self.config.gamma)
                for prototype in prototypes
            )
            if log_every > 0 and ((idx + 1) % log_every == 0 or idx + 1 == total):
                print(
                    f"[SoftDTW_disco] scoring windows: {idx + 1}/{total} "
                    f"(prototypes={len(prototypes)})"
                )
        return starts, scores

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        self.spl_threshold_ = None
        self.spl_loss_buffer_ = []
        self.prototypes_ = None
        super().detect_fit(train_data, test_data)
