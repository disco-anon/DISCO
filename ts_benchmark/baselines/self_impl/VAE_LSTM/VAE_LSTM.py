import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.self_impl.ModernTCN.utils.tools import EarlyStopping
from ts_benchmark.baselines.self_impl.disco import SPLConfig, SPLController
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


DEFAULT_VAE_LSTM_HYPER_PARAMS = {
    "seq_len": 100,
    "batch_size": 128,
    "num_epochs": 10,
    "lr": 1e-3,
    "hidden_dim": 128,
    "latent_dim": 16,
    "num_layers": 1,
    "dropout": 0.0,
    "beta": 1.0,
    "patience": 5,
    "train_val_ratio": 0.7,
    "gradient_clip_norm": 10.0,
    "score_normalize": True,
    "score_source": "last_mse",
    "score_window_agg": "last",
    "score_window_blend": 1.0,
    "score_smooth_window": 1,
    "score_smooth_blend": 0.0,
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


class VAE_LSTMConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_VAE_LSTM_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class VAE_LSTMNet(nn.Module):
    """LSTM sequence VAE for window reconstruction."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output = nn.Linear(hidden_dim, input_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, sample=True):
        batch_size, seq_len, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]
        mean = self.z_mean(h_last)
        logvar = torch.clamp(self.z_logvar(h_last), min=-8.0, max=8.0)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + torch.randn_like(std) * std
        else:
            z = mean

        dec_in = z.unsqueeze(1).repeat(1, seq_len, 1)
        h0_single = torch.tanh(self.z_to_hidden(z)).unsqueeze(0)
        h0 = h0_single.repeat(self.num_layers, 1, 1).contiguous()
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(dec_in, (h0, c0))
        rec = self.output(dec_out)
        return rec, mean, logvar


class VAE_LSTM:
    def __init__(self, **kwargs):
        self.config = VAE_LSTMConfig(**kwargs)
        self.model_name = "VAE_LSTM"
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

    def _build_model(self, input_dim):
        return VAE_LSTMNet(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

    def _frame_from_values(self, values, like_df):
        return pd.DataFrame(values, columns=like_df.columns, index=like_df.index)

    def _fit_transform_train_valid(self, train_data):
        train_df_raw, valid_df_raw = train_val_split(
            train_data, self.config.train_val_ratio, None
        )
        if len(valid_df_raw) < self.config.seq_len:
            valid_df_raw = train_data.tail(min(len(train_data), self.config.seq_len))
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
        rec, mean, logvar = self.model(x, sample=sample)
        mse_t = torch.mean((rec - x).pow(2), dim=2)
        mse = mse_t.sum(dim=1)
        kl = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        loss = mse + self.config.beta * kl
        return loss, mse, kl, mse_t

    def _select_window_score(self, mse_t, mse, loss, source):
        if source == "mean_mse":
            return mse_t.mean(dim=1)
        if source == "max_mse":
            return mse_t.max(dim=1).values
        if source == "sum_mse":
            return mse
        if source == "loss":
            return loss
        if source == "meanmax_mse":
            return 0.5 * mse_t.mean(dim=1) + 0.5 * mse_t.max(dim=1).values
        return mse_t[:, -1]

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
        chunks = []
        for x, _ in loader:
            x = x.float().to(self.device)
            loss, mse, _, mse_t = self._loss_terms(x, sample=False)
            scores = self._select_window_score(
                mse_t, mse, loss, getattr(self.config, "score_source", "last_mse")
            )
            chunks.append(scores.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.array([])

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
        loader = self._make_loader(data, mode="test")
        window_scores = self._compute_window_scores(loader)
        scores = self._last_point_scores_to_timestep(len(data), window_scores)
        window_agg = getattr(self.config, "score_window_agg", "last")
        if window_agg in {"mean", "max"}:
            overlap_scores = self._overlap_scores_to_timestep(
                len(data), window_scores, window_agg
            )
            blend = float(getattr(self.config, "score_window_blend", 1.0))
            blend = min(max(blend, 0.0), 1.0)
            scores = (1.0 - blend) * scores + blend * overlap_scores

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
        normalize = self.config.score_normalize
        self.config.score_normalize = False
        raw_train_scores = self._score_frame(train_df)
        self.config.score_normalize = normalize
        self.train_score_median = np.median(raw_train_scores)
        self.train_score_iqr = (
            np.percentile(raw_train_scores, 75) - np.percentile(raw_train_scores, 25)
        )

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        if len(train_data) < self.config.seq_len:
            raise ValueError(
                f"Training data length must be at least seq_len={self.config.seq_len}."
            )
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
                f"[VAE_LSTM] Epoch {epoch:02d}/{self.config.num_epochs} "
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
        test_scores = self._score_frame(test_df)
        train_scores = (
            self._score_frame(self._train_df_scaled)
            if self._train_df_scaled is not None
            else np.array([])
        )
        combined = np.concatenate([train_scores, test_scores], axis=0)
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


class VAE_LSTM_disco(VAE_LSTM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "VAE_LSTM_disco"
        spl_kwargs = {k: v for k, v in kwargs.items() if k in _SPL_KEYS}
        self.spl_config = SPLConfig(**spl_kwargs)

    def _train_one_epoch_spl(self, loader, optimizer, spl):
        self.model.train()
        total, total_w, steps = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.float().to(self.device)
            loss_per_sample, mse, _, _ = self._loss_terms(x, sample=True)
            source = self.spl_config.spl_difficulty_source
            if source == "loss":
                difficulty = loss_per_sample.detach() - loss_per_sample.detach().min()
            else:
                difficulty = mse.detach()
            loss, avg_w = spl.compute_loss(
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
            total_w += avg_w
            steps += 1
        steps = max(1, steps)
        return total / steps, total_w / steps

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        if len(train_data) < self.config.seq_len:
            raise ValueError(
                f"Training data length must be at least seq_len={self.config.seq_len}."
            )
        train_df, valid_df = self._fit_transform_train_valid(train_data)
        self.model = self._build_model(train_data.shape[1])
        train_loader = self._make_loader(train_df, mode="train")
        valid_loader = self._make_loader(valid_df, mode="val")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
        spl = SPLController(self.spl_config, num_epochs=self.config.num_epochs)

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            phase = spl.on_epoch_start(epoch - 1)
            train_loss, avg_w = self._train_one_epoch_spl(train_loader, optimizer, spl)
            val_loss = self._evaluate(valid_loader)
            print(
                f"[VAE_LSTM_disco] Epoch {epoch:02d}/{self.config.num_epochs} "
                f"[{phase}] Train={train_loss:.6f} Val={val_loss:.6f} "
                f"avg_w={avg_w:.4f} lambda={spl.threshold_scalar:.6f} "
                f"time={time.time() - t0:.1f}s"
            )

            if epoch - 1 == self.spl_config.spl_start_epoch - 1:
                spl.on_warmup_end(val_loss, self.model)
                if self.spl_config.enable_spl:
                    print(
                        f"[VAE_LSTM_disco] SPL warmup baseline saved. "
                        f"Val={val_loss:.6f}"
                    )

            fuse = spl.on_epoch_end(val_loss, self.model)
            if fuse["fused"]:
                print(f"[VAE_LSTM_disco] SPL fuse triggered: {fuse['reason']}")
                early_stopping = EarlyStopping(patience=self.config.patience, verbose=False)
                early_stopping(spl.warmup_vali, self.model)
            else:
                early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break

        self.best_state_dict = copy.deepcopy(early_stopping.check_point)
        self._finalize_fit(train_df)
