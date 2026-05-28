import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from ts_benchmark.baselines.self_impl.ModernTCN.utils.tools import EarlyStopping
from ts_benchmark.baselines.self_impl.disco import SPLConfig, SPLController
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


DEFAULT_OMNI_ANOMALY_PARAMS = {
    "seq_len": 100,
    "batch_size": 50,
    "num_epochs": 10,
    "lr": 1e-3,
    "hidden_dim": 500,
    "z_dim": 3,
    "dense_dim": 500,
    "dropout": 0.0,
    "beta": 1.0,
    "posterior_flow_type": "nf",
    "nf_layers": 20,
    "std_epsilon": 1e-4,
    "patience": 5,
    "train_val_ratio": 0.7,
    "gradient_clip_norm": 10.0,
    "score_normalize": True,
    "score_source": "last_nll",
    "score_source_secondary": None,
    "score_source_blend": 1.0,
    "score_source_fusion": "raw",
    "score_smooth_window": 1,
    "score_smooth_blend": 0.0,
    "score_rollmax_window": 1,
    "score_rollmax_blend": 0.0,
    "score_window_agg": "last",
    "score_window_blend": 1.0,
    "score_variants": None,
    "spl_stop_on_fuse": False,
    "spl_active_lr_scale": 1.0,
    "spl_keep_fused_state": False,
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


class OmniAnomalyConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_OMNI_ANOMALY_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class PlanarFlow(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.u = nn.Parameter(torch.empty(z_dim))
        self.w = nn.Parameter(torch.empty(z_dim))
        self.b = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.u, mean=0.0, std=0.01)
        nn.init.normal_(self.w, mean=0.0, std=0.01)

    def _u_hat(self):
        wu = torch.dot(self.w, self.u)
        m = -1.0 + torch.nn.functional.softplus(wu)
        return self.u + ((m - wu) * self.w) / (self.w.pow(2).sum() + 1e-8)

    def forward(self, z):
        linear = z @ self.w + self.b
        h = torch.tanh(linear)
        u_hat = self._u_hat()
        z_next = z + h.unsqueeze(-1) * u_hat
        psi = (1.0 - h.pow(2)).unsqueeze(-1) * self.w
        det = 1.0 + psi @ u_hat.unsqueeze(-1)
        log_abs_det = torch.log(torch.abs(det.squeeze(-1)) + 1e-8)
        return z_next, log_abs_det


class OmniAnomalyNet(nn.Module):
    """
    PyTorch reproduction of OmniAnomaly's recurrent VAE core.

    The original TensorFlow implementation uses recurrent q(z|x), a connected
    latent prior, and Gaussian p(x|z). This module keeps those semantics:
    q(z_t|x_1:t) is produced by a GRU, p(z_t|z_<t) by a prior GRU, and
    p(x_t|z_1:t) is a diagonal Gaussian decoder.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        z_dim,
        dense_dim,
        dropout=0.0,
        std_epsilon=1e-4,
        posterior_flow_type="nf",
        nf_layers=20,
    ):
        super().__init__()
        self.std_epsilon = std_epsilon
        self.z_dim = z_dim
        self.q_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.q_mean = nn.Linear(hidden_dim, z_dim)
        self.q_logstd = nn.Linear(hidden_dim, z_dim)

        self.pz_rnn = nn.GRU(z_dim, hidden_dim, batch_first=True)
        self.pz_mean = nn.Linear(hidden_dim, z_dim)
        self.pz_logstd = nn.Linear(hidden_dim, z_dim)

        self.px_rnn = nn.GRU(z_dim, hidden_dim, batch_first=True)
        self.px_hidden = nn.Sequential(
            nn.Linear(hidden_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.px_mean = nn.Linear(dense_dim, input_dim)
        self.px_logstd = nn.Linear(dense_dim, input_dim)
        use_nf = posterior_flow_type == "nf" and int(nf_layers) > 0
        self.flows = nn.ModuleList(
            [PlanarFlow(z_dim) for _ in range(int(nf_layers) if use_nf else 0)]
        )

    def _std(self, value):
        return torch.nn.functional.softplus(value) + self.std_epsilon

    def forward(self, x, sample=True):
        q_h, _ = self.q_rnn(x)
        q_mean = self.q_mean(q_h)
        q_std = self._std(self.q_logstd(q_h))
        z0 = q_mean + torch.randn_like(q_std) * q_std if sample else q_mean
        z = z0
        flow_log_det = torch.zeros(z.shape[:2], device=z.device, dtype=z.dtype)
        if self.flows:
            flat_z = z.reshape(-1, self.z_dim)
            flat_log_det = torch.zeros(flat_z.shape[0], device=z.device, dtype=z.dtype)
            for flow in self.flows:
                flat_z, cur_log_det = flow(flat_z)
                flat_log_det = flat_log_det + cur_log_det
            z = flat_z.reshape_as(z)
            flow_log_det = flat_log_det.reshape(z.shape[:2])

        prior_in = torch.zeros_like(z)
        prior_in[:, 1:, :] = z[:, :-1, :]
        pz_h, _ = self.pz_rnn(prior_in)
        pz_mean = self.pz_mean(pz_h)
        pz_std = self._std(self.pz_logstd(pz_h))

        px_h, _ = self.px_rnn(z)
        px_h = self.px_hidden(px_h)
        px_mean = self.px_mean(px_h)
        px_std = self._std(self.px_logstd(px_h))

        return z0, z, q_mean, q_std, pz_mean, pz_std, px_mean, px_std, flow_log_det


def _normal_log_prob(value, mean, std):
    var = std.pow(2)
    return -0.5 * (
        math.log(2.0 * math.pi) + torch.log(var) + (value - mean).pow(2) / var
    )


def _kl_normal(q_mean, q_std, p_mean, p_std):
    return (
        torch.log(p_std / q_std)
        + (q_std.pow(2) + (q_mean - p_mean).pow(2)) / (2.0 * p_std.pow(2))
        - 0.5
    )


class OmniAnomaly:
    def __init__(self, **kwargs):
        self.config = OmniAnomalyConfig(**kwargs)
        self.model_name = "OmniAnomaly"
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_state_dict = None
        self._train_df_scaled = None
        self.train_score_median = None
        self.train_score_iqr = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self):
        return self.model_name

    def _build_model(self, input_dim):
        return OmniAnomalyNet(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            z_dim=self.config.z_dim,
            dense_dim=self.config.dense_dim,
            dropout=self.config.dropout,
            std_epsilon=self.config.std_epsilon,
            posterior_flow_type=self.config.posterior_flow_type,
            nf_layers=self.config.nf_layers,
        ).to(self.device)

    def _frame_from_values(self, values, like_df):
        return pd.DataFrame(values, columns=like_df.columns, index=like_df.index)

    def _fit_transform_train_valid(self, train_data):
        train_df_raw, valid_df_raw = train_val_split(
            train_data, self.config.train_val_ratio, None
        )
        self.scaler.fit(train_df_raw.values)
        train_df = self._frame_from_values(
            self.scaler.transform(train_df_raw.values), train_df_raw
        )
        valid_df = self._frame_from_values(
            self.scaler.transform(valid_df_raw.values), valid_df_raw
        )
        return train_df, valid_df

    def _make_loader(self, data, mode):
        return anomaly_detection_data_provider(
            data,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode=mode,
        )

    def _loss_terms(self, x, sample=True):
        z0, z, q_mean, q_std, pz_mean, pz_std, px_mean, px_std, flow_log_det = self.model(
            x, sample=sample
        )
        log_px = _normal_log_prob(x, px_mean, px_std).sum(dim=2)
        nll_t = -log_px
        log_q0 = _normal_log_prob(z0, q_mean, q_std).sum(dim=2)
        log_pz = _normal_log_prob(z, pz_mean, pz_std).sum(dim=2)
        kl_t = log_q0 - flow_log_det - log_pz
        nll = nll_t.sum(dim=1)
        kl = kl_t.sum(dim=1)
        loss = nll + self.config.beta * kl
        return loss, nll, kl, nll_t[:, -1]

    def _select_window_score(self, nll_t, nll, loss, source):
        if source == "mean_nll":
            return nll_t.mean(dim=1)
        if source == "max_nll":
            return nll_t.max(dim=1).values
        if source == "sum_nll":
            return nll
        if source == "loss":
            return loss
        if source == "meanmax_nll":
            mean_score = nll_t.mean(dim=1)
            max_score = nll_t.max(dim=1).values
            return 0.5 * mean_score + 0.5 * max_score
        return nll_t[:, -1]

    def _window_score_components(self, x):
        z0, z, q_mean, q_std, pz_mean, pz_std, px_mean, px_std, flow_log_det = self.model(
            x, sample=False
        )
        log_px = _normal_log_prob(x, px_mean, px_std).sum(dim=2)
        nll_t = -log_px
        log_q0 = _normal_log_prob(z0, q_mean, q_std).sum(dim=2)
        log_pz = _normal_log_prob(z, pz_mean, pz_std).sum(dim=2)
        kl_t = log_q0 - flow_log_det - log_pz
        nll = nll_t.sum(dim=1)
        kl = kl_t.sum(dim=1)
        loss = nll + self.config.beta * kl
        return nll_t, nll, loss

    def _window_score_terms(self, x):
        nll_t, nll, loss = self._window_score_components(x)
        source = getattr(self.config, "score_source", "last_nll")
        score = self._select_window_score(nll_t, nll, loss, source)
        secondary = getattr(self.config, "score_source_secondary", None)
        if secondary:
            blend = float(getattr(self.config, "score_source_blend", 1.0))
            blend = min(max(blend, 0.0), 1.0)
            score = blend * score + (
                1.0 - blend
            ) * self._select_window_score(nll_t, nll, loss, secondary)
        return score

    def _train_one_epoch(self, loader, optimizer):
        self.model.train()
        total, steps = 0.0, 0
        for x, _ in loader:
            x = x.float().to(self.device)
            loss, _, _, _ = self._loss_terms(x, sample=True)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            optimizer.step()
            total += loss.item()
            steps += 1
        return total / max(1, steps)

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total, steps = 0.0, 0
        for x, _ in loader:
            x = x.float().to(self.device)
            loss, _, _, _ = self._loss_terms(x, sample=False)
            total += loss.mean().item()
            steps += 1
        return total / max(1, steps)

    @torch.no_grad()
    def _compute_window_scores(self, loader):
        self.model.eval()
        secondary = getattr(self.config, "score_source_secondary", None)
        fusion = getattr(self.config, "score_source_fusion", "raw")
        if secondary and fusion in {"rank", "rank_product"}:
            primary_chunks, secondary_chunks = [], []
            source = getattr(self.config, "score_source", "last_nll")
            for x, _ in loader:
                x = x.float().to(self.device)
                nll_t, nll, loss = self._window_score_components(x)
                primary = self._select_window_score(nll_t, nll, loss, source)
                secondary_score = self._select_window_score(nll_t, nll, loss, secondary)
                primary_chunks.append(primary.detach().cpu().numpy())
                secondary_chunks.append(secondary_score.detach().cpu().numpy())
            primary_scores = np.concatenate(primary_chunks, axis=0)
            secondary_scores = np.concatenate(secondary_chunks, axis=0)
            blend = float(getattr(self.config, "score_source_blend", 1.0))
            blend = min(max(blend, 0.0), 1.0)
            primary_rank = self._rank01(primary_scores)
            secondary_rank = self._rank01(secondary_scores)
            if fusion == "rank_product":
                return np.power(primary_rank, blend) * np.power(
                    secondary_rank, 1.0 - blend
                )
            return blend * primary_rank + (1.0 - blend) * secondary_rank

        chunks = []
        for x, _ in loader:
            x = x.float().to(self.device)
            scores = self._window_score_terms(x)
            chunks.append(scores.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _rank01(scores):
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) <= 1:
            return np.ones_like(scores, dtype=np.float64)
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(len(scores), dtype=np.float64)
        return ranks / float(len(scores) - 1)

    def _last_point_scores_to_timestep(self, n_rows, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if n_rows < self.config.seq_len:
            return np.zeros(n_rows, dtype=np.float64)
        fill_value = float(np.min(scores)) if len(scores) else 0.0
        out = np.full(n_rows, fill_value, dtype=np.float64)
        start = self.config.seq_len - 1
        end = min(n_rows, start + len(scores))
        out[start:end] = scores[: end - start]
        return out

    def _overlap_scores_to_timestep(self, n_rows, scores, mode):
        scores = np.asarray(scores, dtype=np.float64)
        if n_rows < self.config.seq_len or len(scores) == 0:
            return np.zeros(n_rows, dtype=np.float64)
        n_windows = min(len(scores), n_rows - self.config.seq_len + 1)
        scores = scores[:n_windows]
        if mode == "max":
            fill_value = float(np.min(scores))
            out = np.full(n_rows, fill_value, dtype=np.float64)
            for i, score in enumerate(scores):
                out[i:i + self.config.seq_len] = np.maximum(
                    out[i:i + self.config.seq_len], score
                )
            return out

        sums = np.zeros(n_rows, dtype=np.float64)
        counts = np.zeros(n_rows, dtype=np.float64)
        for i, score in enumerate(scores):
            sums[i:i + self.config.seq_len] += score
            counts[i:i + self.config.seq_len] += 1.0
        fill_value = float(np.min(scores))
        return np.divide(
            sums,
            counts,
            out=np.full(n_rows, fill_value, dtype=np.float64),
            where=counts > 0,
        )

    def _score_frame(self, data):
        # mode="test" gives step=1 windows; mode="thre" is non-overlapping here.
        loader = self._make_loader(data, mode="test")
        window_scores = self._compute_window_scores(loader)
        scores = self._last_point_scores_to_timestep(len(data), window_scores)
        window_agg = getattr(self.config, "score_window_agg", "last")
        if window_agg in {"mean", "max"}:
            overlap_scores = self._overlap_scores_to_timestep(
                len(data), window_scores, window_agg
            )
            window_blend = float(getattr(self.config, "score_window_blend", 1.0))
            window_blend = min(max(window_blend, 0.0), 1.0)
            scores = (1.0 - window_blend) * scores + window_blend * overlap_scores
        rollmax_window = int(getattr(self.config, "score_rollmax_window", 1))
        rollmax_blend = float(getattr(self.config, "score_rollmax_blend", 0.0))
        if rollmax_window > 1 and rollmax_blend > 0.0:
            rollmax_window = rollmax_window + 1 if rollmax_window % 2 == 0 else rollmax_window
            half = rollmax_window // 2
            padded = np.pad(scores, (half, half), mode="edge")
            rollmax_scores = np.array(
                [np.max(padded[i:i + rollmax_window]) for i in range(len(scores))],
                dtype=np.float64,
            )
            rollmax_blend = min(max(rollmax_blend, 0.0), 1.0)
            scores = (1.0 - rollmax_blend) * scores + rollmax_blend * rollmax_scores
        smooth_window = int(getattr(self.config, "score_smooth_window", 1))
        smooth_blend = float(getattr(self.config, "score_smooth_blend", 0.0))
        if smooth_window > 1 and smooth_blend != 0.0:
            smooth_window = smooth_window + 1 if smooth_window % 2 == 0 else smooth_window
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            smooth_scores = np.convolve(scores, kernel, mode="same")
            smooth_blend = min(max(smooth_blend, -1.0), 1.0)
            scores = (1.0 - smooth_blend) * scores + smooth_blend * smooth_scores
        if (
            self.config.score_normalize
            and self.train_score_median is not None
            and self.train_score_iqr is not None
        ):
            scores = (scores - self.train_score_median) / (self.train_score_iqr + 1e-9)
        return scores

    def _finalize_fit(self, train_df):
        self.model.load_state_dict(self.best_state_dict)
        self._train_df_scaled = train_df
        raw_train_scores = self._score_frame(train_df)
        self.train_score_median = np.median(raw_train_scores)
        self.train_score_iqr = (
            np.percentile(raw_train_scores, 75) - np.percentile(raw_train_scores, 25)
        )

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        train_df, valid_df = self._fit_transform_train_valid(train_data)
        self.model = self._build_model(train_data.shape[1])
        train_loader = self._make_loader(train_df, mode="train")
        valid_loader = self._make_loader(valid_df, mode="val")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, optimizer)
            val_loss = self._evaluate(valid_loader)
            print(
                f"[OmniAnomaly] Epoch {epoch:02d}/{self.config.num_epochs} "
                f"Train={train_loss:.6f} Val={val_loss:.6f} "
                f"time={time.time() - t0:.1f}s"
            )
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break

        self.best_state_dict = copy.deepcopy(early_stopping.check_point)
        self._finalize_fit(train_df)

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        self.model.load_state_dict(self.best_state_dict)
        test_df = self._frame_from_values(self.scaler.transform(test.values), test)
        variants = getattr(self.config, "score_variants", None)
        if variants:
            scores_by_variant = {}
            for idx, variant in enumerate(variants):
                if not isinstance(variant, dict):
                    continue
                name = str(variant.get("name", f"variant_{idx}"))
                overrides = {k: v for k, v in variant.items() if k != "name"}
                saved_values = {k: getattr(self.config, k, None) for k in overrides}
                for key, value in overrides.items():
                    setattr(self.config, key, value)
                scores_by_variant[name] = self._score_frame(test_df)
                for key, value in saved_values.items():
                    setattr(self.config, key, value)
            if scores_by_variant:
                first_scores = next(iter(scores_by_variant.values()))
                return scores_by_variant, first_scores
        scores = self._score_frame(test_df)
        return scores, scores

    def detect_label(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        self.model.load_state_dict(self.best_state_dict)
        test_df = self._frame_from_values(self.scaler.transform(test.values), test)
        train_energy = self._score_frame(self._train_df_scaled)
        test_energy = self._score_frame(test_df)
        combined = np.concatenate([train_energy, test_energy], axis=0)
        ratios = (
            self.config.anomaly_ratio
            if isinstance(self.config.anomaly_ratio, list)
            else [self.config.anomaly_ratio]
        )
        preds = {}
        for ratio in ratios:
            threshold = np.percentile(combined, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)
        return preds, test_energy


class OmniAnomaly_disco(OmniAnomaly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "OmniAnomaly_disco"
        spl_kwargs = {k: v for k, v in kwargs.items() if k in _SPL_KEYS}
        self.spl_config = SPLConfig(**spl_kwargs)

    def _train_one_epoch_spl(self, loader, optimizer, spl):
        self.model.train()
        total, avg_w, steps = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.float().to(self.device)
            loss_per_sample, nll, _, last_nll = self._loss_terms(x, sample=True)
            if self.spl_config.spl_difficulty_source == "nll":
                difficulty = nll.detach()
            elif self.spl_config.spl_difficulty_source == "loss":
                difficulty = loss_per_sample.detach()
            elif self.spl_config.spl_difficulty_source == "last_nll":
                difficulty = last_nll.detach()
            else:
                difficulty = nll.detach()
            difficulty = difficulty - difficulty.min()
            loss, batch_w = spl.compute_loss(loss_per_sample, difficulty=difficulty)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            optimizer.step()
            total += loss.item()
            avg_w += batch_w
            steps += 1
        steps = max(1, steps)
        return total / steps, avg_w / steps

    def _set_optimizer_lr_for_phase(self, optimizer, phase):
        active_scale = float(getattr(self.config, "spl_active_lr_scale", 1.0))
        active_scale = min(max(active_scale, 0.0), 1.0)
        for group in optimizer.param_groups:
            base_lr = group.setdefault("base_lr", group["lr"])
            group["lr"] = base_lr * active_scale if phase == "active" else base_lr

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        train_df, valid_df = self._fit_transform_train_valid(train_data)
        self.model = self._build_model(train_data.shape[1])
        train_loader = self._make_loader(train_df, mode="train")
        valid_loader = self._make_loader(valid_df, mode="val")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
        spl = SPLController(self.spl_config, num_epochs=self.config.num_epochs)
        warmup_best_val = None
        warmup_best_state = None

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            phase = spl.on_epoch_start(epoch - 1)
            self._set_optimizer_lr_for_phase(optimizer, phase)
            train_loss, avg_w = self._train_one_epoch_spl(train_loader, optimizer, spl)
            val_loss = self._evaluate(valid_loader)
            print(
                f"[OmniAnomaly_disco] Epoch {epoch:02d}/{self.config.num_epochs} "
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
            pre_fuse_state = None
            if getattr(self.config, "spl_keep_fused_state", False):
                pre_fuse_state = copy.deepcopy(self.model.state_dict())
            fuse = spl.on_epoch_end(val_loss, self.model)
            if fuse["fused"]:
                print(f"[OmniAnomaly_disco] SPL fuse triggered: {fuse['reason']}")
                early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
                if getattr(self.config, "spl_keep_fused_state", False):
                    self.model.load_state_dict(pre_fuse_state)
                    early_stopping(val_loss, self.model)
                else:
                    early_stopping(spl.warmup_vali, self.model)
                if getattr(self.config, "spl_stop_on_fuse", False):
                    break
            else:
                early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break

        self.best_state_dict = copy.deepcopy(early_stopping.check_point)
        self._finalize_fit(train_df)
