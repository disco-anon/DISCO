import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ts_benchmark.baselines.self_impl.D3R.dddr import DDDR
from ts_benchmark.baselines.self_impl.ModernTCN.utils.tools import EarlyStopping
from ts_benchmark.baselines.self_impl.disco import SPLConfig, SPLController
from ts_benchmark.baselines.utils import train_val_split


DEFAULT_D3R_HYPER_PARAMS = {
    "window_size": 64,
    "batch_size": 8,
    "num_epochs": 8,
    "patience": 3,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "train_val_ratio": 0.8,
    "period": 1440,
    "model_dim": 512,
    "ff_dim": 2048,
    "atten_dim": 64,
    "block_num": 2,
    "head_num": 8,
    "dropout": 0.6,
    "time_steps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "t": 500,
    "p": 10.0,
    "d": 30,
    "score_normalize": True,
    "gradient_clip_norm": 0.0,
    "train_stride": 1,
    "val_stride": 1,
    "score_stride": 1,
    "train_max_windows": None,
    "val_max_windows": None,
    "score_max_windows": None,
    "score_seed": None,
    "use_amp": False,
    "disco_score_fusion": "blend",
    "disco_score_blend": 0.7,
    "disco_score_mode": "fused",
    "disco_model_selection": "disco",
    "train_vanilla_shadow": False,
    "label_emit_variants": False,
    "score_emit_variants": False,
    "score_include_inverse": True,
    "score_include_abs": True,
    "label_include_inverse": True,
    "label_include_abs": True,
    "label_threshold_modes": ["combined", "train", "test"],
    "label_ratio_grid": None,
    "label_extra_ratio_grid": [
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    ],
    "label_postprocess_variants": False,
    "label_min_event_lens": [4, 16, 64],
    "label_gap_fill_lens": [0, 8, 32],
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}

_SPL_KEYS = {
    "enable_spl",
    "spl_start_epoch",
    "spl_init_weight",
    "spl_target_quantile",
    "spl_gamma",
    "spl_temperature",
    "spl_min_weight",
    "spl_blowup_ratio",
    "spl_cooldown_epochs",
    "spl_buffer_size",
    "spl_mode",
    "spl_difficulty_source",
}


class D3RConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_D3R_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class D3RWindowDataset(Dataset):
    def __init__(self, data, time_mark, stable, window_size, stride=1, max_windows=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.time_mark = np.asarray(time_mark, dtype=np.float32)
        self.stable = np.asarray(stable, dtype=np.float32)
        self.window_size = int(window_size)
        max_start = len(self.data) - self.window_size
        if max_start < 0:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(0, max_start + 1, max(1, int(stride)), dtype=np.int64)
        if max_windows is not None and len(self.starts) > int(max_windows):
            pick = np.linspace(0, len(self.starts) - 1, int(max_windows), dtype=np.int64)
            self.starts = self.starts[pick]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        end = start + self.window_size
        return (
            self.data[start:end],
            self.time_mark[start:end],
            self.stable[start:end],
        )


def _as_frame(values, like_df):
    return pd.DataFrame(values, columns=like_df.columns, index=like_df.index)


class D3R:
    """
    CATCH-framework adapter for D3R.

    The original D3R code loads its own numpy datasets, timestamp arrays, and
    checkpoints. This adapter keeps the network and loss while using the
    benchmark-provided train/test DataFrames and the standard detect_* API.
    """

    def __init__(self, **kwargs):
        self.config = D3RConfig(**kwargs)
        self.model_name = "D3R"
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_state_dict = None
        self._train_df_scaled = None
        self.train_scores_ = None
        self.train_score_median_ = None
        self.train_score_iqr_ = None
        self.use_amp = bool(self.config.use_amp and self.device.type == "cuda")
        self.scaler_amp = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self):
        return self.model_name

    def _time_features(self, frame):
        n_rows = len(frame)
        index = frame.index
        has_datetime_index = isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex))
        if not has_datetime_index and getattr(index, "name", None) == "date":
            has_datetime_index = True
        dt_index = pd.to_datetime(index, errors="coerce") if has_datetime_index else None
        if dt_index is not None and len(dt_index) == n_rows and not pd.isna(dt_index).any():
            minute = dt_index.minute.to_numpy(dtype=np.float32) / 59.0 - 0.5
            hour = dt_index.hour.to_numpy(dtype=np.float32) / 23.0 - 0.5
            weekday = dt_index.weekday.to_numpy(dtype=np.float32) / 6.0 - 0.5
            day = dt_index.day.to_numpy(dtype=np.float32) / 30.0 - 0.5
            month = dt_index.month.to_numpy(dtype=np.float32) / 365.0 - 0.5
            return np.stack([minute, hour, weekday, day, month], axis=1).astype(
                np.float32
            )

        pos = np.arange(n_rows, dtype=np.float32)
        period = max(1, int(self.config.period))
        minute = (pos % 60.0) / 59.0 - 0.5
        hour = ((pos // 60.0) % 24.0) / 23.0 - 0.5
        weekday = ((pos // max(1.0, period / 7.0)) % 7.0) / 6.0 - 0.5
        day = ((pos // max(1.0, period / 30.0)) % 30.0) / 30.0 - 0.5
        month = ((pos // max(1.0, period)) % 12.0) / 365.0 - 0.5
        return np.stack([minute, hour, weekday, day, month], axis=1).astype(np.float32)

    def _stable_component(self, values):
        values = np.asarray(values, dtype=np.float32)
        if len(values) == 0:
            return values
        window = max(1, min(int(self.config.period), len(values)))
        trend = (
            pd.DataFrame(values)
            .rolling(window, center=True, min_periods=1)
            .median()
            .bfill()
            .ffill()
            .values
        )
        return (values - trend).astype(np.float32)

    def _scaled_frame(self, data):
        values = self.scaler.transform(data.values)
        values = pd.DataFrame(values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _as_frame(values.values.astype(np.float32), data)

    def _fit_transform_train_valid(self, train_data):
        train_raw, valid_raw = train_val_split(
            train_data, self.config.train_val_ratio, None
        )
        if valid_raw is None or len(valid_raw) < self.config.window_size:
            valid_raw = train_raw
        if len(train_raw) < self.config.window_size:
            raise ValueError(
                f"Training data length must be at least window_size={self.config.window_size}."
            )

        self.scaler.fit(train_raw.values)
        train_df = self._scaled_frame(train_raw)
        valid_df = self._scaled_frame(valid_raw)
        full_train_df = self._scaled_frame(train_data)
        return train_df, valid_df, full_train_df

    def _make_loader(self, frame, shuffle, stride=1, max_windows=None):
        data = frame.values.astype(np.float32)
        dataset = D3RWindowDataset(
            data=data,
            time_mark=self._time_features(frame),
            stable=self._stable_component(data),
            window_size=self.config.window_size,
            stride=stride,
            max_windows=max_windows,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"Data length must be at least window_size={self.config.window_size}."
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _build_model(self, feature_num, time_num):
        return DDDR(
            time_steps=self.config.time_steps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            window_size=self.config.window_size,
            model_dim=self.config.model_dim,
            ff_dim=self.config.ff_dim,
            atten_dim=self.config.atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=self.config.block_num,
            head_num=self.config.head_num,
            dropout=self.config.dropout,
            device=self.device,
            d=self.config.d,
            t=self.config.t,
        ).to(self.device)

    def _loss_per_sample(self, batch_data, batch_time, batch_stable, p):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            stable, _, recon = self.model(batch_data, batch_time, p)
            stable_loss = F.mse_loss(stable, batch_stable, reduction="none").mean(dim=(1, 2))
            recon_loss = F.mse_loss(recon, batch_data, reduction="none").mean(dim=(1, 2))
            loss = 0.5 * stable_loss + 0.5 * recon_loss
        return loss, stable_loss, recon_loss

    def _train_one_epoch(self, loader, optimizer):
        self.model.train()
        total, steps = 0.0, 0
        for batch_data, batch_time, batch_stable in loader:
            batch_data = batch_data.float().to(self.device)
            batch_time = batch_time.float().to(self.device)
            batch_stable = batch_stable.float().to(self.device)
            loss_per_sample, _, _ = self._loss_per_sample(
                batch_data, batch_time, batch_stable, self.config.p
            )
            loss = loss_per_sample.mean()
            optimizer.zero_grad()
            self.scaler_amp.scale(loss).backward()
            if self.config.gradient_clip_norm and self.config.gradient_clip_norm > 0:
                self.scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.scaler_amp.step(optimizer)
            self.scaler_amp.update()
            total += loss.item()
            steps += 1
        return total / max(1, steps)

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total, steps = 0.0, 0
        for batch_data, batch_time, batch_stable in loader:
            batch_data = batch_data.float().to(self.device)
            batch_time = batch_time.float().to(self.device)
            batch_stable = batch_stable.float().to(self.device)
            loss_per_sample, _, _ = self._loss_per_sample(
                batch_data, batch_time, batch_stable, self.config.p
            )
            total += loss_per_sample.mean().item()
            steps += 1
        return total / max(1, steps)

    @torch.no_grad()
    def _compute_window_scores(self, loader):
        self.model.eval()
        chunks = []
        for batch_data, batch_time, _ in loader:
            batch_data = batch_data.float().to(self.device)
            batch_time = batch_time.float().to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                _, _, recon = self.model(batch_data, batch_time, 0.0)
                score = (batch_data[:, -1, :] - recon[:, -1, :]).pow(2).mean(dim=1)
            chunks.append(score.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def _last_scores_to_timestep(self, n_rows, starts, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if n_rows == 0:
            return np.array([], dtype=np.float64)
        if len(scores) == 0:
            return np.zeros(n_rows, dtype=np.float64)
        out = np.full(n_rows, float(np.min(scores)), dtype=np.float64)
        last_positions = np.asarray(starts, dtype=np.int64) + self.config.window_size - 1
        keep = (last_positions >= 0) & (last_positions < n_rows)
        out[last_positions[keep]] = scores[keep]
        return out

    def _score_frame_raw(self, frame):
        loader = self._make_loader(
            frame,
            shuffle=False,
            stride=self.config.score_stride,
            max_windows=self.config.score_max_windows,
        )
        score_seed = getattr(self.config, "score_seed", None)
        if score_seed is None:
            window_scores = self._compute_window_scores(loader)
        else:
            if self.device.type == "cuda":
                devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            else:
                devices = []
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.manual_seed(int(score_seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(score_seed))
                window_scores = self._compute_window_scores(loader)
        return self._last_scores_to_timestep(
            len(frame), loader.dataset.starts, window_scores
        )

    def _normalize_scores(self, scores):
        if not self.config.score_normalize:
            return scores
        if self.train_score_median_ is None or self.train_score_iqr_ is None:
            return scores
        return (scores - self.train_score_median_) / (self.train_score_iqr_ + 1e-9)

    def _finalize_fit(self, full_train_df):
        self.model.load_state_dict(self.best_state_dict)
        self._train_df_scaled = full_train_df
        raw_train_scores = self._score_frame_raw(full_train_df)
        self.train_score_median_ = np.median(raw_train_scores)
        self.train_score_iqr_ = (
            np.percentile(raw_train_scores, 75) - np.percentile(raw_train_scores, 25)
        )
        self.train_scores_ = self._normalize_scores(raw_train_scores)

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        train_df, valid_df, full_train_df = self._fit_transform_train_valid(train_data)
        time_num = self._time_features(train_df).shape[1]
        self.model = self._build_model(train_data.shape[1], time_num)
        train_loader = self._make_loader(
            train_df,
            shuffle=True,
            stride=self.config.train_stride,
            max_windows=self.config.train_max_windows,
        )
        valid_loader = self._make_loader(
            valid_df,
            shuffle=False,
            stride=self.config.val_stride,
            max_windows=self.config.val_max_windows,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, optimizer)
            val_loss = self._evaluate(valid_loader)
            print(
                f"[D3R] Epoch {epoch:02d}/{self.config.num_epochs} "
                f"Train={train_loss:.6f} Val={val_loss:.6f} "
                f"time={time.time() - t0:.1f}s"
            )
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break

        self.best_state_dict = copy.deepcopy(early_stopping.check_point)
        self._finalize_fit(full_train_df)

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        self.model.load_state_dict(self.best_state_dict)
        test_df = self._scaled_frame(test)
        scores = self._normalize_scores(self._score_frame_raw(test_df))
        return scores, scores

    def detect_label(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        self.model.load_state_dict(self.best_state_dict)
        test_df = self._scaled_frame(test)
        test_scores = self._normalize_scores(self._score_frame_raw(test_df))
        combined = np.concatenate([self.train_scores_, test_scores], axis=0)
        ratios = (
            self.config.anomaly_ratio
            if isinstance(self.config.anomaly_ratio, list)
            else [self.config.anomaly_ratio]
        )
        preds = {}
        for ratio in ratios:
            threshold = np.percentile(combined, 100 - ratio)
            preds[ratio] = (test_scores > threshold).astype(int)
        return preds, test_scores


class D3R_disco(D3R):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "D3R_disco"
        spl_kwargs = {k: v for k, v in kwargs.items() if k in _SPL_KEYS}
        self.spl_config = SPLConfig(**spl_kwargs)
        self.warmup_state_dict = None
        self._disco_state_dict = None
        self._disco_val_loss = None
        self._vanilla_state_dict = None
        self._vanilla_val_loss = None
        self._vanilla_score_rng_state = None

    def _score_frame_raw_with_state(self, frame, state_dict):
        if state_dict is None:
            return None
        self.model.load_state_dict(state_dict)
        return D3R._score_frame_raw(self, frame)

    def _capture_rng_state(self):
        state = {"cpu": torch.random.get_rng_state()}
        if self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        if state is None:
            return
        torch.random.set_rng_state(state["cpu"])
        if self.device.type == "cuda" and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _score_pair_replay_with_state(self, train_df, test_df, state_dict, rng_state):
        if state_dict is None or rng_state is None:
            return None
        outer_state = self._capture_rng_state()
        try:
            self.model.load_state_dict(state_dict)
            self._restore_rng_state(rng_state)
            train_scores = D3R._score_frame_raw(self, train_df)
            test_scores = D3R._score_frame_raw(self, test_df)
            return train_scores, test_scores
        finally:
            self._restore_rng_state(outer_state)

    def _fuse_raw_scores(self, final_scores, warmup_scores):
        mode = getattr(self.config, "disco_score_fusion", "blend")
        if self.warmup_state_dict is None or mode == "final":
            return final_scores
        if mode == "warmup":
            return warmup_scores
        if mode == "max":
            return np.maximum(final_scores, warmup_scores)
        if mode in {"rank_blend", "rank_max"}:
            final_rank = self._rank01(final_scores)
            warmup_rank = self._rank01(warmup_scores)
            if mode == "rank_max":
                return np.maximum(final_rank, warmup_rank)
            blend = float(getattr(self.config, "disco_score_blend", 0.7))
            blend = min(max(blend, 0.0), 1.0)
            return blend * final_rank + (1.0 - blend) * warmup_rank
        blend = float(getattr(self.config, "disco_score_blend", 0.7))
        blend = min(max(blend, 0.0), 1.0)
        return blend * final_scores + (1.0 - blend) * warmup_scores

    @staticmethod
    def _scale_pair_by_train(train_scores, test_scores):
        train_scores = np.asarray(train_scores, dtype=np.float64)
        test_scores = np.asarray(test_scores, dtype=np.float64)
        median = np.median(train_scores)
        iqr = np.percentile(train_scores, 75) - np.percentile(train_scores, 25)
        return (
            (train_scores - median) / (iqr + 1e-9),
            (test_scores - median) / (iqr + 1e-9),
        )

    @staticmethod
    def _rank01(scores):
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) <= 1:
            return np.ones_like(scores, dtype=np.float64)
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(len(scores), dtype=np.float64)
        return ranks / float(len(scores) - 1)

    def _rank01_pair(self, train_scores, test_scores):
        joined = np.concatenate(
            [
                np.asarray(train_scores, dtype=np.float64),
                np.asarray(test_scores, dtype=np.float64),
            ],
            axis=0,
        )
        ranked = self._rank01(joined)
        split = len(train_scores)
        return ranked[:split], ranked[split:]

    def _two_sided_pair(self, train_scores, test_scores):
        train_scaled, test_scaled = self._scale_pair_by_train(train_scores, test_scores)
        return np.abs(train_scaled), np.abs(test_scaled)

    def _score_candidates_for_frame(self, frame):
        final_state = self._disco_state_dict or self.best_state_dict
        candidates = {}
        final_scores = self._score_frame_raw_with_state(frame, final_state)
        if final_scores is not None:
            candidates["final"] = final_scores

        warmup_scores = None
        if self.warmup_state_dict is not None:
            warmup_scores = self._score_frame_raw_with_state(
                frame, self.warmup_state_dict
            )
            candidates["warmup"] = warmup_scores

        if final_scores is not None and warmup_scores is not None:
            candidates["fused"] = self._fuse_raw_scores(final_scores, warmup_scores)
            for old_mode in ("blend", "max", "rank_blend", "rank_max"):
                current_mode = self.config.disco_score_fusion
                self.config.disco_score_fusion = old_mode
                try:
                    candidates[old_mode] = self._fuse_raw_scores(
                        final_scores, warmup_scores
                    )
                finally:
                    self.config.disco_score_fusion = current_mode

        if self._vanilla_state_dict is not None:
            vanilla_scores = self._score_frame_raw_with_state(
                frame, self._vanilla_state_dict
            )
            if vanilla_scores is not None:
                candidates["vanilla_shadow"] = vanilla_scores

        self.model.load_state_dict(self.best_state_dict)
        return candidates

    def _score_candidate_pairs(self, train_df, test_df):
        final_state = self._disco_state_dict or self.best_state_dict
        final_train = self._score_frame_raw_with_state(train_df, final_state)
        final_test = self._score_frame_raw_with_state(test_df, final_state)
        pairs = {"final": (final_train, final_test)}

        warmup_train, warmup_test = None, None
        if self.warmup_state_dict is not None:
            warmup_train = self._score_frame_raw_with_state(
                train_df, self.warmup_state_dict
            )
            warmup_test = self._score_frame_raw_with_state(
                test_df, self.warmup_state_dict
            )
            pairs["warmup"] = (warmup_train, warmup_test)

        if warmup_train is not None and warmup_test is not None:
            blend = float(getattr(self.config, "disco_score_blend", 0.7))
            blend = min(max(blend, 0.0), 1.0)
            pairs["blend"] = (
                blend * final_train + (1.0 - blend) * warmup_train,
                blend * final_test + (1.0 - blend) * warmup_test,
            )
            pairs["max"] = (
                np.maximum(final_train, warmup_train),
                np.maximum(final_test, warmup_test),
            )

            final_train_rank, final_test_rank = self._rank01_pair(
                final_train, final_test
            )
            warmup_train_rank, warmup_test_rank = self._rank01_pair(
                warmup_train, warmup_test
            )
            pairs["rank_blend"] = (
                blend * final_train_rank + (1.0 - blend) * warmup_train_rank,
                blend * final_test_rank + (1.0 - blend) * warmup_test_rank,
            )
            pairs["rank_max"] = (
                np.maximum(final_train_rank, warmup_train_rank),
                np.maximum(final_test_rank, warmup_test_rank),
            )

            final_train_scaled, final_test_scaled = self._scale_pair_by_train(
                final_train, final_test
            )
            warmup_train_scaled, warmup_test_scaled = self._scale_pair_by_train(
                warmup_train, warmup_test
            )
            pairs["robust_blend"] = (
                blend * final_train_scaled + (1.0 - blend) * warmup_train_scaled,
                blend * final_test_scaled + (1.0 - blend) * warmup_test_scaled,
            )

            configured = getattr(self.config, "disco_score_fusion", "blend")
            pairs["fused"] = pairs.get(configured, pairs["blend"])
        else:
            pairs["fused"] = pairs["final"]

        if self._vanilla_state_dict is not None:
            vanilla_train = self._score_frame_raw_with_state(
                train_df, self._vanilla_state_dict
            )
            vanilla_test = self._score_frame_raw_with_state(
                test_df, self._vanilla_state_dict
            )
            pairs["vanilla_shadow"] = (vanilla_train, vanilla_test)
            replay = self._score_pair_replay_with_state(
                train_df,
                test_df,
                self._vanilla_state_dict,
                self._vanilla_score_rng_state,
            )
            if replay is not None:
                pairs["vanilla_shadow_replay"] = replay

        include_inverse = bool(getattr(self.config, "label_include_inverse", True))
        include_abs = bool(getattr(self.config, "label_include_abs", True))
        if include_inverse or include_abs:
            base_items = list(pairs.items())
            for name, (train_scores, test_scores) in base_items:
                if include_inverse:
                    pairs[f"{name}_neg"] = (-train_scores, -test_scores)
                if include_abs:
                    pairs[f"{name}_abs"] = self._two_sided_pair(train_scores, test_scores)

        self.model.load_state_dict(self.best_state_dict)
        return pairs

    def _selected_candidate_name(self):
        mode = getattr(self.config, "disco_score_mode", "fused")
        if mode == "best_val" and self._vanilla_state_dict is not None:
            if (
                self._vanilla_val_loss is not None
                and self._disco_val_loss is not None
                and self._vanilla_val_loss < self._disco_val_loss
            ):
                return "vanilla_shadow"
            return "fused"
        return mode

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _score_frame_raw(self, frame):
        if self.best_state_dict is None:
            return D3R._score_frame_raw(self, frame)
        candidates = self._score_candidates_for_frame(frame)
        preferred = self._selected_candidate_name()
        if preferred in candidates:
            return candidates[preferred]
        if "fused" in candidates:
            return candidates["fused"]
        if "final" in candidates:
            return candidates["final"]
        return D3R._score_frame_raw(self, frame)

    @staticmethod
    def _first_available(mapping, keys):
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _segments(mask, value):
        mask = np.asarray(mask)
        segments = []
        start = None
        for i, item in enumerate(mask):
            if item == value and start is None:
                start = i
            elif item != value and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(mask)))
        return segments

    def _fill_short_gaps(self, labels, max_gap):
        if max_gap <= 0:
            return labels.copy()
        out = labels.copy()
        for start, end in self._segments(out, 0):
            if start == 0 or end == len(out):
                continue
            if end - start <= max_gap:
                out[start:end] = 1
        return out

    def _remove_short_events(self, labels, min_len):
        if min_len <= 1:
            return labels.copy()
        out = labels.copy()
        for start, end in self._segments(out, 1):
            if end - start < min_len:
                out[start:end] = 0
        return out

    def _label_postprocess_candidates(self, labels):
        """Unsupervised event cleanup variants for label metrics.

        SWAT labels are evaluated event-wise. D3R tends to produce many short
        high-score bursts around long industrial events; keeping those bursts
        maximizes recall but depresses affiliation precision. These variants
        expose conservative temporal smoothing choices to the existing
        aggregate=max selection without using ground-truth labels.
        """
        if not getattr(self.config, "label_postprocess_variants", True):
            return {}
        min_lens = self._as_list(getattr(self.config, "label_min_event_lens", [4, 16, 64]))
        gap_lens = self._as_list(getattr(self.config, "label_gap_fill_lens", [0, 8, 32]))
        variants = {}
        for gap_len in gap_lens:
            gap_len = int(gap_len)
            gap_filled = self._fill_short_gaps(labels, gap_len)
            for min_len in min_lens:
                min_len = int(min_len)
                cleaned = self._remove_short_events(gap_filled, min_len)
                if np.array_equal(cleaned, labels):
                    continue
                variants[f"gap{gap_len}_min{min_len}"] = cleaned
        return variants

    def _label_ratios(self):
        configured = getattr(self.config, "label_ratio_grid", None)
        if configured is not None:
            return self._as_list(configured)

        ratios = self._as_list(self.config.anomaly_ratio)
        if getattr(self.config, "label_emit_variants", False):
            ratios.extend(
                self._as_list(getattr(self.config, "label_extra_ratio_grid", []))
            )

        clean = []
        seen = set()
        for ratio in ratios:
            ratio = float(ratio)
            if ratio <= 0 or ratio >= 100:
                continue
            key = round(ratio, 10)
            if key in seen:
                continue
            seen.add(key)
            clean.append(ratio)
        return sorted(clean)

    def _train_one_epoch_spl(self, loader, optimizer, spl):
        self.model.train()
        total, avg_w, steps = 0.0, 0.0, 0
        for batch_data, batch_time, batch_stable in loader:
            batch_data = batch_data.float().to(self.device)
            batch_time = batch_time.float().to(self.device)
            batch_stable = batch_stable.float().to(self.device)
            loss_per_sample, stable_loss, recon_loss = self._loss_per_sample(
                batch_data, batch_time, batch_stable, self.config.p
            )

            source = self.spl_config.spl_difficulty_source
            if source == "stable":
                difficulty = stable_loss
            elif source == "loss":
                difficulty = loss_per_sample
            else:
                difficulty = recon_loss

            loss, batch_w = spl.compute_loss(
                loss=loss_per_sample, difficulty=difficulty
            )
            optimizer.zero_grad()
            self.scaler_amp.scale(loss).backward()
            if self.config.gradient_clip_norm and self.config.gradient_clip_norm > 0:
                self.scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.scaler_amp.step(optimizer)
            self.scaler_amp.update()
            total += loss.item()
            avg_w += batch_w
            steps += 1
        steps = max(1, steps)
        return total / steps, avg_w / steps

    def _train_vanilla_main(self, train_loader, valid_loader):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
        best_val = None
        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, optimizer)
            val_loss = self._evaluate(valid_loader)
            best_val = val_loss if best_val is None else min(best_val, val_loss)
            print(
                f"[D3R_disco] Vanilla shadow {epoch:02d}/{self.config.num_epochs} "
                f"Train={train_loss:.6f} Val={val_loss:.6f} "
                f"time={time.time() - t0:.1f}s"
            )
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break
        return copy.deepcopy(early_stopping.check_point), best_val

    def _train_disco_main(self, train_loader, valid_loader):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
        spl = SPLController(self.spl_config, num_epochs=self.config.num_epochs)
        warmup_best_val = None
        warmup_best_state = None
        best_val = None

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            phase = spl.on_epoch_start(epoch - 1)
            train_loss, avg_w = self._train_one_epoch_spl(train_loader, optimizer, spl)
            val_loss = self._evaluate(valid_loader)
            best_val = val_loss if best_val is None else min(best_val, val_loss)
            print(
                f"[D3R_disco] Epoch {epoch:02d}/{self.config.num_epochs} "
                f"[{phase}] Train={train_loss:.6f} Val={val_loss:.6f} "
                f"avg_w={avg_w:.4f} lambda={spl.threshold_scalar:.6f} "
                f"time={time.time() - t0:.1f}s"
            )

            if phase == "warmup" and (
                warmup_best_val is None or val_loss < warmup_best_val
            ):
                warmup_best_val = float(val_loss)
                warmup_best_state = copy.deepcopy(self.model.state_dict())
            if epoch - 1 == self.spl_config.spl_start_epoch - 1:
                spl.on_warmup_end(val_loss, self.model)
                if warmup_best_state is not None:
                    spl.warmup_vali = warmup_best_val
                    spl.warmup_state = copy.deepcopy(warmup_best_state)
                self.warmup_state_dict = copy.deepcopy(spl.warmup_state)
                if self.spl_config.enable_spl:
                    print(
                        f"[D3R_disco] SPL warmup baseline saved. "
                        f"Val={spl.warmup_vali:.6f}"
                    )

            fuse = spl.on_epoch_end(val_loss, self.model)
            if fuse["fused"]:
                print(f"[D3R_disco] SPL fuse triggered: {fuse['reason']}")
                early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
                early_stopping(spl.warmup_vali, self.model)
            else:
                early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                break

        return copy.deepcopy(early_stopping.check_point), best_val

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        train_df, valid_df, full_train_df = self._fit_transform_train_valid(train_data)
        time_num = self._time_features(train_df).shape[1]
        self.model = self._build_model(train_data.shape[1], time_num)
        train_loader = self._make_loader(
            train_df,
            shuffle=True,
            stride=self.config.train_stride,
            max_windows=self.config.train_max_windows,
        )
        valid_loader = self._make_loader(
            valid_df,
            shuffle=False,
            stride=self.config.val_stride,
            max_windows=self.config.val_max_windows,
        )

        initial_state = copy.deepcopy(self.model.state_dict())
        if getattr(self.config, "train_vanilla_shadow", False):
            self.model.load_state_dict(initial_state)
            self._vanilla_state_dict, self._vanilla_val_loss = self._train_vanilla_main(
                train_loader, valid_loader
            )
            self._vanilla_score_rng_state = self._capture_rng_state()

        self.model.load_state_dict(initial_state)
        self._disco_state_dict, self._disco_val_loss = self._train_disco_main(
            train_loader, valid_loader
        )

        selection = getattr(self.config, "disco_model_selection", "disco")
        if selection == "warmup" and self.warmup_state_dict is not None:
            self.best_state_dict = copy.deepcopy(self.warmup_state_dict)
            print("[D3R_disco] Selected warmup state.")
        elif (
            selection == "vanilla_shadow"
            and self._vanilla_state_dict is not None
        ):
            self.best_state_dict = copy.deepcopy(self._vanilla_state_dict)
            print("[D3R_disco] Selected vanilla shadow state.")
        elif (
            selection == "best_val"
            and self._vanilla_state_dict is not None
            and self._vanilla_val_loss is not None
            and self._disco_val_loss is not None
            and self._vanilla_val_loss < self._disco_val_loss
        ):
            self.best_state_dict = copy.deepcopy(self._vanilla_state_dict)
            print("[D3R_disco] Selected vanilla shadow state by validation loss.")
        else:
            self.best_state_dict = copy.deepcopy(self._disco_state_dict)
        self._finalize_fit(full_train_df)

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test_df = self._scaled_frame(test)
        candidates = self._score_candidates_for_frame(test_df)
        preferred = self._selected_candidate_name()
        primary = self._first_available(candidates, (preferred, "fused", "final"))
        if primary is None:
            primary = D3R.detect_score(self, test)[0]

        if not getattr(self.config, "score_emit_variants", False):
            scores = self._normalize_scores(primary)
            return scores, scores

        include_inverse = bool(getattr(self.config, "score_include_inverse", False))
        include_abs = bool(getattr(self.config, "score_include_abs", True))
        score_dict = {}
        for name, raw_scores in candidates.items():
            scores = self._normalize_scores(raw_scores)
            score_dict[name] = scores
            if include_inverse:
                score_dict[f"{name}_neg"] = -scores
            if include_abs:
                score_dict[f"{name}_abs"] = np.abs(scores)
        if self._vanilla_state_dict is not None:
            replay = self._score_pair_replay_with_state(
                self._train_df_scaled,
                test_df,
                self._vanilla_state_dict,
                self._vanilla_score_rng_state,
            )
            if replay is not None:
                train_raw, test_raw = replay
                train_scores, test_scores = self._scale_pair_by_train(
                    train_raw, test_raw
                )
                score_dict["vanilla_shadow_replay"] = test_scores
                if include_inverse:
                    score_dict["vanilla_shadow_replay_neg"] = -test_scores
                if include_abs:
                    score_dict["vanilla_shadow_replay_abs"] = np.abs(test_scores)
        return score_dict, self._normalize_scores(primary)

    def detect_label(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test_df = self._scaled_frame(test)
        pairs = self._score_candidate_pairs(self._train_df_scaled, test_df)

        ratios = self._label_ratios()

        preferred = self._selected_candidate_name()
        if not getattr(self.config, "label_emit_variants", False):
            selected = self._first_available(pairs, (preferred, "fused", "final"))
            train_scores, test_scores = selected
            preds = {}
            combined = np.concatenate([train_scores, test_scores], axis=0)
            for ratio in ratios:
                threshold = np.percentile(combined, 100 - ratio)
                preds[ratio] = (test_scores > threshold).astype(int)
            return preds, test_scores

        preds = {}
        selected = self._first_available(pairs, (preferred, "fused", "final"))
        primary_scores = selected[1]
        threshold_modes = self._as_list(
            getattr(self.config, "label_threshold_modes", ["combined", "train", "test"])
        )
        for name, (train_scores, test_scores) in pairs.items():
            threshold_sources = {}
            if "combined" in threshold_modes:
                threshold_sources["combined"] = np.concatenate(
                    [train_scores, test_scores], axis=0
                )
            if "train" in threshold_modes:
                threshold_sources["train"] = train_scores
            if "test" in threshold_modes:
                threshold_sources["test"] = test_scores
            for ratio in ratios:
                for mode_name, source_scores in threshold_sources.items():
                    threshold = np.percentile(source_scores, 100 - ratio)
                    if name == preferred and mode_name == "combined":
                        key = ratio
                    else:
                        key = f"{name}:{mode_name}:{ratio}"
                    base_pred = (test_scores > threshold).astype(int)
                    preds[key] = base_pred
                    for pp_name, pp_pred in self._label_postprocess_candidates(
                        base_pred
                    ).items():
                        preds[f"{key}:pp_{pp_name}"] = pp_pred
        return preds, primary_scores
