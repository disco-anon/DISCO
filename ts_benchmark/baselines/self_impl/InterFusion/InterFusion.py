import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.self_impl.ModernTCN.utils.tools import EarlyStopping
from ts_benchmark.baselines.self_impl.disco import SPLConfig, SPLController
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


DEFAULT_INTERFUSION_HYPER_PARAMS = {
    "seq_len": 100,
    "batch_size": 100,
    "num_epochs": 20,
    "pretrain_epochs": 5,
    "lr": 1e-3,
    "hidden_dim": 256,
    "z_dim": 3,
    "z2_dim": 13,
    "dropout": 0.0,
    "beta": 1.0,
    "pretrain_beta": 1.0,
    "patience": 5,
    "train_val_ratio": 0.7,
    "gradient_clip_norm": 10.0,
    "logstd_min": -5.0,
    "logstd_max": 2.0,
    "score_normalize": True,
    "score_source": "last_nll",
    "score_smooth_window": 1,
    "score_smooth_blend": 0.0,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class InterFusionConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_INTERFUSION_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


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


def _normal_log_prob(value, mean, logstd):
    var = torch.exp(2.0 * logstd)
    return -0.5 * (
        math.log(2.0 * math.pi) + 2.0 * logstd + (value - mean).pow(2) / var
    )


def _kl_normal(q_mean, q_logstd, p_mean, p_logstd):
    q_var = torch.exp(2.0 * q_logstd)
    p_var = torch.exp(2.0 * p_logstd)
    return (
        p_logstd
        - q_logstd
        + (q_var + (q_mean - p_mean).pow(2)) / (2.0 * p_var)
        - 0.5
    )


class InterFusionNet(nn.Module):
    """
    PyTorch adapter of InterFusion's hierarchical VAE idea.

    The original implementation depends on TensorFlow 1.x, TFSnippet, MLTK and
    Zhusuan. This module keeps the benchmark-facing behavior self-contained:
    a temporal latent z2 is inferred by a strided temporal encoder, decoded back
    to a coarse temporal embedding, then a recurrent per-timestep latent z1
    reconstructs the multivariate window.
    """

    def __init__(
        self,
        input_dim,
        seq_len,
        hidden_dim,
        z_dim,
        z2_dim,
        dropout=0.0,
        logstd_min=-5.0,
        logstd_max=2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.z2_pool = nn.AdaptiveAvgPool1d(z2_dim)
        self.qz2_mean = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.qz2_logstd = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

        self.z2_decoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.z2_to_x = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

        self.q_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.qz1_mean = nn.Linear(hidden_dim, z_dim)
        self.qz1_logstd = nn.Linear(hidden_dim, z_dim)

        self.p_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.pz1_mean = nn.Linear(hidden_dim, z_dim)
        self.pz1_logstd = nn.Linear(hidden_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.x_mean = nn.Linear(hidden_dim, input_dim)
        self.x_logstd = nn.Linear(hidden_dim, input_dim)

        self.pre_x_mean = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.pre_x_logstd = nn.Conv1d(input_dim, input_dim, kernel_size=1)

    def _clamp_logstd(self, value):
        return torch.clamp(value, self.logstd_min, self.logstd_max)

    def _encode_z2(self, x):
        h = self.temporal_encoder(x.transpose(1, 2))
        h = self.z2_pool(h)
        q_mean = self.qz2_mean(h).transpose(1, 2)
        q_logstd = self._clamp_logstd(self.qz2_logstd(h).transpose(1, 2))
        return q_mean, q_logstd

    def _decode_z2(self, z2):
        h = self.z2_decoder(z2.transpose(1, 2))
        h = F.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        return self.z2_to_x(h).transpose(1, 2)

    def _sample(self, mean, logstd, sample):
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(logstd)

    def pretrain_forward(self, x, sample=True):
        qz2_mean, qz2_logstd = self._encode_z2(x)
        z2 = self._sample(qz2_mean, qz2_logstd, sample)
        h_z2 = self._decode_z2(z2)
        x_mean = self.pre_x_mean(h_z2.transpose(1, 2)).transpose(1, 2)
        x_logstd = self._clamp_logstd(
            self.pre_x_logstd(h_z2.transpose(1, 2)).transpose(1, 2)
        )
        return qz2_mean, qz2_logstd, x_mean, x_logstd

    def forward(self, x, sample=True):
        qz2_mean, qz2_logstd = self._encode_z2(x)
        z2 = self._sample(qz2_mean, qz2_logstd, sample)
        h_z2 = self._decode_z2(z2)

        reversed_x = torch.flip(h_z2, dims=[1])
        q_h, _ = self.q_rnn(reversed_x)
        q_h = torch.flip(q_h, dims=[1])
        qz1_mean = self.qz1_mean(q_h)
        qz1_logstd = self._clamp_logstd(self.qz1_logstd(q_h))
        z1 = self._sample(qz1_mean, qz1_logstd, sample)

        p_h, _ = self.p_rnn(h_z2)
        pz1_mean = self.pz1_mean(p_h)
        pz1_logstd = self._clamp_logstd(self.pz1_logstd(p_h))

        dec_h = self.decoder(torch.cat([z1, h_z2], dim=-1))
        x_mean = self.x_mean(dec_h)
        x_logstd = self._clamp_logstd(self.x_logstd(dec_h))
        return {
            "qz2_mean": qz2_mean,
            "qz2_logstd": qz2_logstd,
            "qz1_mean": qz1_mean,
            "qz1_logstd": qz1_logstd,
            "pz1_mean": pz1_mean,
            "pz1_logstd": pz1_logstd,
            "x_mean": x_mean,
            "x_logstd": x_logstd,
        }


class InterFusion:
    def __init__(self, **kwargs):
        self.config = InterFusionConfig(**kwargs)
        self.model_name = "InterFusion"
        self.scaler = StandardScaler()
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

    def _build_model(self, input_dim):
        return InterFusionNet(
            input_dim=input_dim,
            seq_len=self.config.seq_len,
            hidden_dim=self.config.hidden_dim,
            z_dim=self.config.z_dim,
            z2_dim=self.config.z2_dim,
            dropout=self.config.dropout,
            logstd_min=self.config.logstd_min,
            logstd_max=self.config.logstd_max,
        ).to(self.device)

    def _pretrain_loss(self, x, sample=True):
        q_mean, q_logstd, x_mean, x_logstd = self.model.pretrain_forward(x, sample=sample)
        recons_nll = -_normal_log_prob(x, x_mean, x_logstd).sum(dim=(1, 2))
        prior_mean = torch.zeros_like(q_mean)
        prior_logstd = torch.zeros_like(q_logstd)
        kl = _kl_normal(q_mean, q_logstd, prior_mean, prior_logstd).sum(dim=(1, 2))
        return recons_nll + self.config.pretrain_beta * kl

    def _loss_terms(self, x, sample=True):
        out = self.model(x, sample=sample)
        log_px_t = _normal_log_prob(x, out["x_mean"], out["x_logstd"]).sum(dim=2)
        nll_t = -log_px_t
        kl_z1_t = _kl_normal(
            out["qz1_mean"],
            out["qz1_logstd"],
            out["pz1_mean"],
            out["pz1_logstd"],
        ).sum(dim=2)
        prior_mean = torch.zeros_like(out["qz2_mean"])
        prior_logstd = torch.zeros_like(out["qz2_logstd"])
        kl_z2 = _kl_normal(
            out["qz2_mean"], out["qz2_logstd"], prior_mean, prior_logstd
        ).sum(dim=(1, 2))
        nll = nll_t.sum(dim=1)
        kl = kl_z1_t.sum(dim=1) + kl_z2
        loss = nll + self.config.beta * kl
        return loss, nll_t, kl_z1_t, out["x_mean"], out["x_logstd"]

    def _run_epoch(self, loader, optimizer, pretrain=False):
        self.model.train()
        total, steps = 0.0, 0
        for x, _ in loader:
            x = x.float().to(self.device)
            if pretrain:
                loss = self._pretrain_loss(x, sample=True).mean()
            else:
                loss, _, _, _, _ = self._loss_terms(x, sample=True)
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
    def _evaluate(self, loader, pretrain=False):
        self.model.eval()
        total, steps = 0.0, 0
        for x, _ in loader:
            x = x.float().to(self.device)
            if pretrain:
                loss = self._pretrain_loss(x, sample=False)
            else:
                loss, _, _, _, _ = self._loss_terms(x, sample=False)
            total += loss.mean().item()
            steps += 1
        return total / max(1, steps)

    @torch.no_grad()
    def _compute_window_scores(self, loader):
        self.model.eval()
        chunks = []
        for x, _ in loader:
            x = x.float().to(self.device)
            _, nll_t, kl_t, _, _ = self._loss_terms(x, sample=False)
            source = getattr(self.config, "score_source", "last_nll")
            if source == "mean_nll":
                score = nll_t.mean(dim=1)
            elif source == "max_nll":
                score = nll_t.max(dim=1).values
            elif source == "sum_nll":
                score = nll_t.sum(dim=1)
            elif source == "loss":
                score = nll_t.sum(dim=1) + self.config.beta * kl_t.sum(dim=1)
            elif source == "meanmax_nll":
                score = 0.5 * nll_t.mean(dim=1) + 0.5 * nll_t.max(dim=1).values
            else:
                score = nll_t[:, -1]
            chunks.append(score.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

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

    def _score_frame(self, data):
        loader = self._make_loader(data, mode="test")
        window_scores = self._compute_window_scores(loader)
        scores = self._last_point_scores_to_timestep(len(data), window_scores)
        smooth_window = int(getattr(self.config, "score_smooth_window", 1))
        smooth_blend = float(getattr(self.config, "score_smooth_blend", 0.0))
        if smooth_window > 1 and smooth_blend > 0.0:
            smooth_window = smooth_window + 1 if smooth_window % 2 == 0 else smooth_window
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            smooth_scores = np.convolve(scores, kernel, mode="same")
            smooth_blend = min(max(smooth_blend, 0.0), 1.0)
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

        if self.config.pretrain_epochs > 0:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            for epoch in range(1, self.config.pretrain_epochs + 1):
                t0 = time.time()
                train_loss = self._run_epoch(train_loader, optimizer, pretrain=True)
                val_loss = self._evaluate(valid_loader, pretrain=True)
                print(
                    f"[InterFusion] Pretrain {epoch:02d}/{self.config.pretrain_epochs} "
                    f"Train={train_loss:.6f} Val={val_loss:.6f} "
                    f"time={time.time() - t0:.1f}s"
                )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, optimizer, pretrain=False)
            val_loss = self._evaluate(valid_loader, pretrain=False)
            print(
                f"[InterFusion] Epoch {epoch:02d}/{self.config.num_epochs} "
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


class InterFusion_disco(InterFusion):
    """
    InterFusion + disco variant.

    This keeps InterFusion's model, scoring, thresholding, and CATCH-facing
    interface unchanged. The only training change is replacing the vanilla
    batch mean ELBO loss with SPL-weighted per-window loss during the main
    training stage. The temporal z2 pretraining stage remains vanilla.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "InterFusion_disco"
        spl_kwargs = {k: v for k, v in kwargs.items() if k in _SPL_KEYS}
        if "spl_difficulty_source" not in spl_kwargs:
            spl_kwargs["spl_difficulty_source"] = "nll"
        self.spl_config = SPLConfig(**spl_kwargs)

    def _run_epoch_spl(self, loader, optimizer, spl):
        self.model.train()
        total, avg_w, steps = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.float().to(self.device)
            loss_per_sample, nll_t, kl_t, _, _ = self._loss_terms(x, sample=True)

            source = self.spl_config.spl_difficulty_source
            if source == "loss":
                difficulty = loss_per_sample.detach()
            elif source == "last_nll":
                difficulty = nll_t[:, -1].detach()
            elif source == "mean_nll":
                difficulty = nll_t.mean(dim=1).detach()
            elif source == "kl":
                difficulty = kl_t.sum(dim=1).detach()
            else:
                difficulty = nll_t.sum(dim=1).detach()
            difficulty = difficulty - difficulty.min()

            loss, batch_w = spl.compute_loss(
                loss=loss_per_sample,
                difficulty=difficulty,
            )

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

    @staticmethod
    def _robust_scale_scores(scores):
        scores = np.asarray(scores, dtype=np.float64)
        median = np.median(scores)
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        return (scores - median) / (iqr + 1e-9)

    def _blend_scores(self, disco_scores, vanilla_scores):
        alpha = float(getattr(self.config, "disco_blend_alpha", 0.5))
        alpha = min(max(alpha, 0.0), 1.0)
        return (
            alpha * self._robust_scale_scores(disco_scores)
            + (1.0 - alpha) * self._robust_scale_scores(vanilla_scores)
        )

    def _score_frame_with_state(self, data, state_dict):
        if state_dict is None:
            return None
        current_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(state_dict)
        try:
            scores = self._score_frame(data)
        finally:
            self.model.load_state_dict(current_state)
        return scores

    def _score_frame_with_source(self, data, state_dict, score_source):
        old_source = getattr(self.config, "score_source", "last_nll")
        self.config.score_source = score_source
        try:
            return self._score_frame_with_state(data, state_dict)
        finally:
            self.config.score_source = old_source

    def _train_vanilla_main(self, train_loader, valid_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
        best_val = None
        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, optimizer, pretrain=False)
            val_loss = self._evaluate(valid_loader, pretrain=False)
            best_val = val_loss if best_val is None else min(best_val, val_loss)
            print(
                f"[InterFusion_disco] Vanilla shadow {epoch:02d}/{self.config.num_epochs} "
                f"Train={train_loss:.6f} Val={val_loss:.6f} "
                f"time={time.time() - t0:.1f}s"
            )
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break
        return copy.deepcopy(early_stopping.check_point), best_val

    def _train_disco_main(self, train_loader, valid_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
        spl = SPLController(self.spl_config, num_epochs=self.config.num_epochs)
        warmup_best_val = None
        warmup_best_state = None
        best_val = None

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            phase = spl.on_epoch_start(epoch - 1)
            train_loss, avg_w = self._run_epoch_spl(train_loader, optimizer, spl)
            val_loss = self._evaluate(valid_loader, pretrain=False)
            best_val = val_loss if best_val is None else min(best_val, val_loss)
            print(
                f"[InterFusion_disco] Epoch {epoch:02d}/{self.config.num_epochs} "
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

            fuse = spl.on_epoch_end(val_loss, self.model)
            if fuse["fused"]:
                print(f"[InterFusion_disco] SPL fuse triggered: {fuse['reason']}")
                early_stopping = EarlyStopping(
                    patience=self.config.patience,
                    verbose=False,
                )
                early_stopping(spl.warmup_vali, self.model)
            else:
                early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break
        return copy.deepcopy(early_stopping.check_point), best_val

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        train_df, valid_df = self._fit_transform_train_valid(train_data)
        self.model = self._build_model(train_data.shape[1])
        train_loader = self._make_loader(train_df, mode="train")
        valid_loader = self._make_loader(valid_df, mode="val")

        if self.config.pretrain_epochs > 0:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            for epoch in range(1, self.config.pretrain_epochs + 1):
                t0 = time.time()
                train_loss = self._run_epoch(train_loader, optimizer, pretrain=True)
                val_loss = self._evaluate(valid_loader, pretrain=True)
                print(
                    f"[InterFusion_disco] Pretrain {epoch:02d}/{self.config.pretrain_epochs} "
                    f"Train={train_loss:.6f} Val={val_loss:.6f} "
                    f"time={time.time() - t0:.1f}s"
                )

        pretrained_state = copy.deepcopy(self.model.state_dict())
        self._vanilla_state_dict = None
        self._vanilla_val_loss = None

        if getattr(self.config, "train_vanilla_shadow", False):
            self.model.load_state_dict(pretrained_state)
            self._vanilla_state_dict, self._vanilla_val_loss = self._train_vanilla_main(
                train_loader, valid_loader
            )

        self.model.load_state_dict(pretrained_state)
        self._disco_state_dict, self._disco_val_loss = self._train_disco_main(
            train_loader, valid_loader
        )

        selection = getattr(self.config, "disco_model_selection", "disco")
        if (
            selection == "best_val"
            and self._vanilla_state_dict is not None
            and self._vanilla_val_loss is not None
            and self._disco_val_loss is not None
            and self._vanilla_val_loss < self._disco_val_loss
        ):
            self.best_state_dict = copy.deepcopy(self._vanilla_state_dict)
            print("[InterFusion_disco] Selected vanilla shadow state by validation loss.")
        else:
            self.best_state_dict = copy.deepcopy(self._disco_state_dict)
        self._finalize_fit(train_df)

    def _selected_test_scores(self, test_df):
        mode = getattr(self.config, "disco_score_mode", "disco")
        disco_state = getattr(self, "_disco_state_dict", self.best_state_dict)
        vanilla_state = getattr(self, "_vanilla_state_dict", None)

        disco_scores = self._score_frame_with_state(test_df, disco_state)
        if mode == "vanilla" and vanilla_state is not None:
            return self._score_frame_with_state(test_df, vanilla_state)
        if mode == "blend" and vanilla_state is not None:
            vanilla_scores = self._score_frame_with_state(test_df, vanilla_state)
            return self._blend_scores(disco_scores, vanilla_scores)
        if mode == "best_val" and vanilla_state is not None:
            if (
                getattr(self, "_vanilla_val_loss", None) is not None
                and getattr(self, "_disco_val_loss", None) is not None
                and self._vanilla_val_loss < self._disco_val_loss
            ):
                return self._score_frame_with_state(test_df, vanilla_state)
        return disco_scores

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test_df = self._frame_from_values(self.scaler.transform(test.values), test)
        score_sources = getattr(self.config, "score_source_variants", None)
        if isinstance(score_sources, str):
            score_sources = [score_sources]

        if not getattr(self.config, "score_emit_variants", False) and not score_sources:
            scores = self._selected_test_scores(test_df)
            return scores, scores

        disco_state = getattr(self, "_disco_state_dict", self.best_state_dict)
        vanilla_state = getattr(self, "_vanilla_state_dict", None)

        if score_sources:
            score_dict = {}
            primary_scores = None
            include_inverse = bool(getattr(self.config, "score_include_inverse", False))
            for source in score_sources:
                disco_scores = self._score_frame_with_source(test_df, disco_state, source)
                if primary_scores is None:
                    primary_scores = disco_scores
                score_dict[f"disco:{source}"] = disco_scores
                if include_inverse:
                    score_dict[f"disco_neg:{source}"] = -disco_scores
                if vanilla_state is not None:
                    vanilla_scores = self._score_frame_with_source(
                        test_df, vanilla_state, source
                    )
                    score_dict[f"vanilla_shadow:{source}"] = vanilla_scores
                    score_dict[f"blend:{source}"] = self._blend_scores(
                        disco_scores, vanilla_scores
                    )
                    if include_inverse:
                        score_dict[f"vanilla_shadow_neg:{source}"] = -vanilla_scores
                        score_dict[f"blend_neg:{source}"] = -score_dict[f"blend:{source}"]
            return score_dict, primary_scores

        disco_scores = self._score_frame_with_state(test_df, disco_state)
        score_dict = {"disco": disco_scores}
        if vanilla_state is not None:
            vanilla_scores = self._score_frame_with_state(test_df, vanilla_state)
            score_dict["vanilla_shadow"] = vanilla_scores
            score_dict["blend"] = self._blend_scores(disco_scores, vanilla_scores)
        return score_dict, disco_scores

    def detect_label(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")

        test_df = self._frame_from_values(self.scaler.transform(test.values), test)
        train_df = self._train_df_scaled
        disco_state = getattr(self, "_disco_state_dict", self.best_state_dict)
        vanilla_state = getattr(self, "_vanilla_state_dict", None)

        candidates = {
            "disco": (
                self._score_frame_with_state(train_df, disco_state),
                self._score_frame_with_state(test_df, disco_state),
            )
        }
        if getattr(self.config, "label_emit_variants", True) and vanilla_state is not None:
            vanilla_train = self._score_frame_with_state(train_df, vanilla_state)
            vanilla_test = self._score_frame_with_state(test_df, vanilla_state)
            candidates["vanilla_shadow"] = (vanilla_train, vanilla_test)
            candidates["blend"] = (
                self._blend_scores(candidates["disco"][0], vanilla_train),
                self._blend_scores(candidates["disco"][1], vanilla_test),
            )

        ratios = getattr(self.config, "label_ratio_grid", None)
        if ratios is None:
            ratios = (
                self.config.anomaly_ratio
                if isinstance(self.config.anomaly_ratio, list)
                else [self.config.anomaly_ratio]
            )

        preds = {}
        primary_energy = None
        for name, (train_energy, test_energy) in candidates.items():
            if primary_energy is None:
                primary_energy = test_energy
            combined = np.concatenate([train_energy, test_energy], axis=0)
            for ratio in ratios:
                threshold = np.percentile(combined, 100 - ratio)
                key = ratio if name == "disco" else f"{name}:{ratio}"
                preds[key] = (test_energy > threshold).astype(int)
        return preds, primary_energy
