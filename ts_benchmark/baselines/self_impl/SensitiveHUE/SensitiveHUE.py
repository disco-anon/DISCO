import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.utils import (
    anomaly_detection_data_provider,
    train_val_split,
)
from ts_benchmark.baselines.self_impl.ModernTCN.utils.tools import EarlyStopping
from ts_benchmark.baselines.self_impl.SensitiveHUE.models.SensitiveHUE_model import (
    SensitiveHUEModel,
)


DEFAULT_SENSITIVE_HUE_HYPER_PARAMS = {
    "seq_len": 24,
    "batch_size": 256,
    "num_epochs": 30,
    "lr": 1e-3,
    "dim_model": 128,
    "head_num": 4,
    "dim_hidden_fc": 256,
    "encode_layer_num": 1,
    "dropout": 0.1,
    "alpha": 1.0,
    "use_prob": True,
    "patience": 10,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "iqr_eps": 1e-9,
    "train_val_ratio": 0.8,
}


class SensitiveHUEConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_SENSITIVE_HUE_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class SensitiveHUE:
    """
    CATCH-framework adapter for SensitiveHUE (ICLR submission, rebuttal copy at
    /root/project/disco_no/SensitiveHUE-master).

    Preserves the model's core: Transformer encoder + heteroscedastic (rec, log_var_recip)
    heads; MTS-NLL loss; training-set IQR-normalize + max-over-channels scoring.

    Follows CATCH conventions for data loading and evaluation:
      - pandas DataFrame input through ``anomaly_detection_data_provider``
      - ``detect_score`` returns per-timestep scores of length T
      - ``detect_label`` returns a dict of {anomaly_ratio: binary labels} from percentile
        thresholds over the combined train+test energy distribution
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.config = SensitiveHUEConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = self.config.seq_len
        self.model_name = "SensitiveHUE"

        self.model = None
        self.best_state_dict = None
        self.train_median = None
        self.train_iqr = None
        self._train_df_scaled = None

        self._mse_loss = nn.MSELoss(reduction="none")

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def _loss_func(self, rec, x, log_var_recip, with_weight=True):
        rec_loss = self._mse_loss(rec, x)
        sigma_loss = rec_loss * log_var_recip.exp() - log_var_recip
        if with_weight:
            var = (-log_var_recip).exp().detach()
            mean_var = var.mean(dim=(0, 1)) ** self.config.alpha
            loss = (var * sigma_loss / mean_var).mean()
        else:
            loss = sigma_loss.mean()
        return rec_loss.mean(), loss

    def _train_one_epoch(self, data_loader, optimizer):
        self.model.train()
        sum_rec, sum_prob, steps = 0.0, 0.0, 0
        for x, _ in data_loader:
            x = x.float().to(self.device)
            rec, log_var_recip = self.model(x)
            rec_loss, prob_loss = self._loss_func(rec, x, log_var_recip, with_weight=True)
            loss = prob_loss if self.config.use_prob else rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_rec += rec_loss.item()
            sum_prob += prob_loss.item()
            steps += 1
        steps = max(1, steps)
        return sum_rec / steps, sum_prob / steps

    @torch.no_grad()
    def _evaluate(self, data_loader):
        self.model.eval()
        sum_rec, sum_prob, steps = 0.0, 0.0, 0
        for x, _ in data_loader:
            x = x.float().to(self.device)
            rec, log_var_recip = self.model(x)
            rec_loss, prob_loss = self._loss_func(rec, x, log_var_recip, with_weight=False)
            sum_rec += rec_loss.item()
            sum_prob += prob_loss.item()
            steps += 1
        steps = max(1, steps)
        return sum_rec / steps, sum_prob / steps

    @torch.no_grad()
    def _compute_anomaly_per_channel(self, data_loader):
        """Run the model over a loader and return per-timestep per-channel anomaly scores
        concatenated across all windows/batches. Shape: (T_total, C)."""
        self.model.eval()
        chunks = []
        for x, _ in data_loader:
            x = x.float().to(self.device)
            rec, log_var_recip = self.model(x)
            score = self._mse_loss(rec, x)
            if self.config.use_prob:
                score = score * log_var_recip.exp() - log_var_recip
            arr = score.detach().cpu().numpy()
            chunks.append(arr.reshape(-1, arr.shape[-1]))
        return np.concatenate(chunks, axis=0)

    def _reduce_channels(self, per_channel_scores):
        """Apply training-set IQR normalization then max-over-channels.
        (T, C) -> (T,)"""
        normalized = (per_channel_scores - self.train_median) / (
            self.train_iqr + self.config.iqr_eps
        )
        return normalized.max(axis=1)

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        config = self.config

        train_df_raw, valid_df_raw = train_val_split(train_data, config.train_val_ratio, None)

        self.scaler.fit(train_df_raw.values)
        train_df = pd.DataFrame(
            self.scaler.transform(train_df_raw.values),
            columns=train_df_raw.columns,
            index=train_df_raw.index,
        )
        valid_df = pd.DataFrame(
            self.scaler.transform(valid_df_raw.values),
            columns=valid_df_raw.columns,
            index=valid_df_raw.index,
        )

        f_in = train_data.shape[1]
        self.model = SensitiveHUEModel(
            step_num_in=config.seq_len,
            f_in=f_in,
            dim_model=config.dim_model,
            head_num=config.head_num,
            dim_hidden_fc=config.dim_hidden_fc,
            encode_layer_num=config.encode_layer_num,
            dropout=config.dropout,
        ).to(self.device)

        train_loader = anomaly_detection_data_provider(
            train_df,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )
        val_loader = anomaly_detection_data_provider(
            valid_df,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        early_stopping = EarlyStopping(patience=config.patience, verbose=False)

        for epoch in range(1, config.num_epochs + 1):
            t0 = time.time()
            train_rec, train_prob = self._train_one_epoch(train_loader, optimizer)
            val_rec, val_prob = self._evaluate(val_loader)
            val_monitor = val_prob if config.use_prob else val_rec
            print(
                f"[SensitiveHUE] Epoch {epoch:02d}/{config.num_epochs} "
                f"Train(rec={train_rec:.6f}, prob={train_prob:.6f}) "
                f"Val(rec={val_rec:.6f}, prob={val_prob:.6f}) "
                f"time={time.time() - t0:.1f}s"
            )
            early_stopping(val_monitor, self.model)
            if early_stopping.early_stop:
                print(f"[SensitiveHUE] Early stopping at epoch {epoch}")
                break

        self.best_state_dict = early_stopping.check_point
        self.model.load_state_dict(self.best_state_dict)
        self._train_df_scaled = train_df

        train_thre_loader = anomaly_detection_data_provider(
            train_df,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )
        train_scores = self._compute_anomaly_per_channel(train_thre_loader)
        self.train_median = np.median(train_scores, axis=0)
        self.train_iqr = iqr(train_scores, axis=0)

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        config = self.config

        test_df = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.best_state_dict)
        self.model.to(self.device).eval()

        thre_loader = anomaly_detection_data_provider(
            test_df,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )
        per_ch = self._compute_anomaly_per_channel(thre_loader)
        scores = self._reduce_channels(per_ch)
        return scores, scores

    def detect_label(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        config = self.config

        test_df = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.best_state_dict)
        self.model.to(self.device).eval()

        train_loader = anomaly_detection_data_provider(
            self._train_df_scaled,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )
        train_per_ch = self._compute_anomaly_per_channel(train_loader)
        train_energy = self._reduce_channels(train_per_ch)

        test_loader = anomaly_detection_data_provider(
            test_df,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )
        test_overlap_per_ch = self._compute_anomaly_per_channel(test_loader)
        test_overlap_energy = self._reduce_channels(test_overlap_per_ch)

        combined_energy = np.concatenate([train_energy, test_overlap_energy], axis=0)

        thre_loader = anomaly_detection_data_provider(
            test_df,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )
        final_per_ch = self._compute_anomaly_per_channel(thre_loader)
        test_energy = self._reduce_channels(final_per_ch)

        ratios = (
            config.anomaly_ratio
            if isinstance(config.anomaly_ratio, list)
            else [config.anomaly_ratio]
        )
        preds = {}
        for ratio in ratios:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)
        return preds, test_energy
