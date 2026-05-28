"""
SensitiveHUE + disco (Self-Paced Learning) variant.

Subclasses ``SensitiveHUE`` and injects a model-agnostic SPL controller into
the training loop. Inherits inference (``detect_score``/``detect_label``),
IQR+max channel aggregation, scaler, and all other behavior unchanged.

Training loss form (matches vanilla SensitiveHUE's ``loss_func(with_weight=True)``
on a per-window basis, then SPL re-weights at the sample level):

    rec_elem    = mse(rec, x)                           # (B, S, C)
    nll_elem    = rec_elem * exp(log_var_recip) - log_var_recip
    var         = exp(-log_var_recip).detach()
    mean_var    = var.mean(dim=(0,1)) ** alpha
    weighted_nll_elem = var * nll_elem / mean_var       # vanilla form
    loss_per_sample   = weighted_nll_elem.mean(dim=(1,2))   # (B,) for SPL

When SPL is inactive (warmup / cooldown / disabled), SPLController returns
``loss_per_sample.mean()``, which equals vanilla's ``loss_func(with_weight=True)``
exactly — guaranteeing disco never under-performs vanilla due to a different
loss formulation.

The MSE per-sample (≥0) is fed as the SPL *difficulty* signal so the Winsorize
clip ``99th_percentile × 10`` makes sense; the α-weighted NLL is the *loss*
that gradients flow through.

See ``disco/spl.py`` for the SPL formulas (ported from CATCH).
"""
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import iqr

from ts_benchmark.baselines.utils import (
    anomaly_detection_data_provider,
    train_val_split,
)
from ts_benchmark.baselines.self_impl.ModernTCN.utils.tools import EarlyStopping
from ts_benchmark.baselines.self_impl.SensitiveHUE.SensitiveHUE import SensitiveHUE
from ts_benchmark.baselines.self_impl.SensitiveHUE.models.SensitiveHUE_model import (
    SensitiveHUEModel,
)
from ts_benchmark.baselines.self_impl.disco import SPLConfig, SPLController


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


class SensitiveHUE_disco(SensitiveHUE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "SensitiveHUE+disco"
        spl_kwargs = {k: v for k, v in kwargs.items() if k in _SPL_KEYS}
        self.spl_config = SPLConfig(**spl_kwargs)

    def _train_one_epoch_spl(self, data_loader, optimizer, spl):
        """SPL-weighted training pass.

        Loss form mirrors vanilla SensitiveHUE's ``loss_func(with_weight=True)``
        per element, reduced to per-sample (B,) for SPL. When SPL is inactive,
        ``compute_loss`` returns ``loss_per_sample.mean()`` which is identical
        to vanilla's reduction — so disco never deviates from vanilla due to a
        different loss formulation; only SPL's sample re-weighting is added
        when active.

        Difficulty signal is selected by ``spl_difficulty_source`` config:
          - ``"mse"`` (default): per-sample MSE (≥ 0 by construction).
          - ``"nll"``: per-sample α-weighted NLL, batch-shifted by its min so
            the result is ≥ 0 (NLL can be negative under SensitiveHUE's
            log-precision parameterization). Aligns SPL's ranking with the
            actual gradient signal.
        Both forms preserve the relative-ordering property SPL's quantile/
        sigmoid operate on; the ≥ 0 invariant keeps spl.py's Winsorize clip
        ``99th_percentile × 10`` meaningful.
        """
        self.model.train()
        sum_rec, sum_nll, sum_w, steps = 0.0, 0.0, 0.0, 0
        for x, _ in data_loader:
            x = x.float().to(self.device)
            rec, log_var_recip = self.model(x)

            rec_elem = self._mse_loss(rec, x)                       # (B, S, C)
            mse_per_sample = rec_elem.mean(dim=(1, 2))              # (B,), ≥ 0

            if self.config.use_prob:
                # vanilla loss_func(with_weight=True), per-element
                nll_elem = rec_elem * log_var_recip.exp() - log_var_recip
                var = (-log_var_recip).exp().detach()
                mean_var = var.mean(dim=(0, 1)) ** self.config.alpha
                weighted_nll_elem = var * nll_elem / mean_var
                loss_per_sample = weighted_nll_elem.mean(dim=(1, 2))  # (B,)
            else:
                loss_per_sample = mse_per_sample

            if self.spl_config.spl_difficulty_source == "nll":
                # batch-shift NLL to ≥ 0; ordering preserved within the batch
                nll_det = loss_per_sample.detach()
                difficulty = nll_det - nll_det.min()
            else:  # "mse" — current default, unchanged behavior
                difficulty = mse_per_sample

            loss, avg_w = spl.compute_loss(
                loss=loss_per_sample, difficulty=difficulty
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_rec += mse_per_sample.detach().mean().item()
            sum_nll += loss_per_sample.detach().mean().item()
            sum_w += avg_w
            steps += 1
        steps = max(1, steps)
        return sum_rec / steps, sum_nll / steps, sum_w / steps

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        config = self.config

        train_df_raw, valid_df_raw = train_val_split(
            train_data, config.train_val_ratio, None
        )
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
        spl = SPLController(self.spl_config, num_epochs=config.num_epochs)

        for epoch in range(1, config.num_epochs + 1):
            t0 = time.time()
            phase = spl.on_epoch_start(epoch - 1)  # 0-indexed

            train_rec, train_nll, train_w = self._train_one_epoch_spl(
                train_loader, optimizer, spl
            )
            val_rec, val_prob = self._evaluate(val_loader)
            val_monitor = val_prob if config.use_prob else val_rec

            print(
                f"[SensitiveHUE+disco] Epoch {epoch:02d}/{config.num_epochs} "
                f"[{phase}] Train(rec={train_rec:.6f}, nll={train_nll:.6f}, "
                f"avg_w={train_w:.4f}) Val(rec={val_rec:.6f}, prob={val_prob:.6f}) "
                f"λ={spl.threshold_scalar:.6f} time={time.time() - t0:.1f}s"
            )

            # Use val_rec (MSE, always >= 0) for SPL fuse — NLL-based val_prob
            # can be negative, which breaks the `val > ratio * baseline` check.
            # save warmup baseline at the last warmup epoch (epoch == spl_start_epoch)
            if epoch - 1 == self.spl_config.spl_start_epoch - 1:
                spl.on_warmup_end(val_rec, self.model)
                if self.spl_config.enable_spl:
                    print(
                        f"[SensitiveHUE+disco] SPL warmup baseline saved. "
                        f"Val(rec)={val_rec:.6f}"
                    )

            fuse = spl.on_epoch_end(val_rec, self.model)
            if fuse["fused"]:
                print(
                    f"[SensitiveHUE+disco] SPL FUSED ({fuse['reason']}). "
                    f"Rolled back to warmup weights; SPL disabled."
                )
                # re-seed EarlyStopping with warmup as current-best, per CATCH's convention
                early_stopping = EarlyStopping(patience=config.patience, verbose=False)
                early_stopping(spl.warmup_vali, self.model)
            else:
                early_stopping(val_monitor, self.model)

            if early_stopping.early_stop:
                print(f"[SensitiveHUE+disco] Early stopping at epoch {epoch}")
                break

        self.best_state_dict = early_stopping.check_point
        self.model.load_state_dict(self.best_state_dict)
        self._train_df_scaled = train_df

        # training-set IQR+max statistics (identical to base class)
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
