import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

from ts_benchmark.baselines.catch.models.CATCH_model import (
    CATCHModel,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss, frequency_criterion
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate

# 新增自步学习相关超参
DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "lr": 0.0001,
    "Mlr": 0.00001,
    "e_layers": 3,
    "n_heads": 2,
    "cf_dim": 64,
    "d_ff": 256,
    "d_model": 128,
    "head_dim": 64,
    "individual": 0,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "auxi_lambda": 0.005,
    "score_lambda": 0.05,
    "regular_lambda": 0.5,
    "temperature": 0.07,
    "patch_stride": 8,
    "patch_size": 16,
    "inference_patch_stride": 1,
    "inference_patch_size": 32,
    "dc_lambda": 0.005,
    "module_first": True,
    "mask": False,
    "pretrained_model": None,
    "num_epochs": 3,
    "batch_size": 128,
    "patience": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "seq_len": 192,
    "pct_start": 0.3,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "lradj": "type1",
    # 自步学习新增参数
    "enable_spl": True,          # 是否启用 SPL
    "spl_init_weight": 0.5,      # 初始阈值对应的分位数（0.5 = 中位数）
    "spl_target_quantile": 0.95, # 训练末期目标分位数
    "spl_temperature": 1.0,      # sigmoid 温度系数（乘以 buffer.std() 自适应缩放）
    "spl_gamma": 0.9,            # 阈值 EMA 系数，越大越平滑
    "spl_start_epoch": 1,        # 前 N 个 epoch 不做 SPL，用于 warmup 和填充 buffer
    "spl_blowup_ratio": 2.0,     # 熔断阈值（vali > 2x warmup 才回退，避免误杀短期抖动）
    "spl_min_weight": 0.3,       # 权重地板：即使最难样本也保留 30% 梯度（保住排序能力/AUC）
    "spl_cooldown_epochs": 1,    # 最后 N 个 epoch 关闭 SPL，恢复尾部细粒度（AUC 更好）
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


class CATCH:
    def __init__(self, **kwargs):
        super(CATCH, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss(reduction='none')  # 修改为none，方便计算每个样本损失
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len

        # 自步学习初始化
        self.spl_threshold = None       # 当前阈值 (EMA)
        self.spl_epoch = 0              # 当前 epoch
        self.spl_loss_buffer = []       # 滑动窗口，存最近若干 batch 的 per-sample loss
        self._spl_buffer_size = 2048    # 缓冲区目标样本数，用于稳定分位数估计
        self.spl_warmup_vali = None     # warmup 结束时的 vali loss（熔断基线）
        self.spl_warmup_state = None    # warmup 结束时的模型权重（熔断回退）
        self.spl_disabled = False       # 熔断后自动关闭 SPL

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.to(device)

                output, _, _ = self.model(input)
                output = output[:, :, :]
                output = output.detach().cpu()
                true = input.detach().cpu()

                loss = criterion(output, true).mean(dim=(1,2)).detach().cpu().numpy()  # 每个样本的损失
                total_loss.append(loss)

        total_loss = np.mean(np.concatenate(total_loss))
        self.model.train()
        return total_loss

    def _spl_compute_loss(self, rec_loss, auxi_loss, dcloss):
        """
        Self-Paced Learning 损失（v3）：
        关键稳定性措施：
        - Winsorize：每次加入 buffer 前把极端 loss 截断到 99 分位（防污染统计）
        - 固定温度尺度：用 warmup 末期的 loss_std 作为温度分母，不再实时更新
          （避免"权重失真→loss 爆炸→std 爆炸→温度爆炸"的正反馈）
        - 熔断支持：detect_fit 可以在 vali loss 恶化时设 spl_disabled=True，
          之后自动退化为普通加权平均
        """
        # --- 1) per-sample scalar loss（不含 dcloss） ---
        rec_loss_sample = rec_loss.mean(dim=(1, 2))
        auxi_loss_sample = (
            auxi_loss.mean(dim=(1, 2)) if auxi_loss.dim() > 1 else auxi_loss
        )
        per_sample = rec_loss_sample + self.config.auxi_lambda * auxi_loss_sample

        # --- 2) Winsorize 后更新滑动缓冲区 ---
        # 仅当 buffer 已有一定体量（>1 个 batch）时才做截断，避免初期压扁数据分布。
        # 截断到 10x 99 分位数只是个"极端离群点"安全阀，不干扰正常的 SPL 动态。
        per_sample_detached = per_sample.detach()
        if len(self.spl_loss_buffer) >= 2:
            buf_cat = torch.cat(self.spl_loss_buffer)
            clip_hi = torch.quantile(buf_cat, 0.99).detach() * 10.0
            per_sample_detached = torch.clamp(per_sample_detached, max=clip_hi)
        self.spl_loss_buffer.append(per_sample_detached)
        max_batches = max(1, self._spl_buffer_size // per_sample.shape[0])
        while len(self.spl_loss_buffer) > max_batches:
            self.spl_loss_buffer.pop(0)

        dc_term = (
            self.config.dc_lambda * dcloss if isinstance(dcloss, torch.Tensor) else 0.0
        )

        # --- 3) Warmup / SPL 未启用 / 已熔断 / Cooldown：普通平均损失 ---
        # 自适应 cooldown：只有训练足够长（warmup + SPL >= 3 epoch）时才开启 cooldown。
        # 否则 cooldown 会挤占本就短的 SPL 窗口（如 PSM num_epochs=3）。
        effective_cooldown = self.config.spl_cooldown_epochs
        spl_budget = self.config.num_epochs - self.config.spl_start_epoch
        if spl_budget <= 2:
            effective_cooldown = 0
        cooldown_start = self.config.num_epochs - effective_cooldown
        in_cooldown = self.spl_epoch >= cooldown_start
        spl_active = (self.config.enable_spl
                      and not self.spl_disabled
                      and not in_cooldown
                      and self.spl_epoch >= self.config.spl_start_epoch)
        if not spl_active:
            loss = per_sample.mean() + dc_term
            return loss, 1.0

        buffer = torch.cat(self.spl_loss_buffer)

        # --- 4) SPL 首次激活时，用完整 buffer 初始化阈值 ---
        if self.spl_threshold is None:
            self.spl_threshold = torch.quantile(
                buffer, self.config.spl_init_weight
            ).detach()

        # --- 5) 目标分位数按训练进度单调增长 ---
        active_epochs = max(1, self.config.num_epochs - self.config.spl_start_epoch)
        progress = (self.spl_epoch - self.config.spl_start_epoch) / active_epochs
        progress = min(max(progress, 0.0), 1.0)
        target_q = (self.config.spl_init_weight
                    + (self.config.spl_target_quantile - self.config.spl_init_weight)
                    * progress)
        target_q = min(max(target_q, 0.05), 0.99)
        target_thr = torch.quantile(buffer, target_q).detach()

        # EMA 更新阈值（γ 大 → 平滑）
        self.spl_threshold = (
            self.config.spl_gamma * self.spl_threshold
            + (1.0 - self.config.spl_gamma) * target_thr
        ).detach()

        # --- 6) 自适应温度：用 buffer 的 std 归一化 sigmoid 斜率 ---
        loss_std = buffer.std().detach() + 1e-8
        scaled_temp = self.config.spl_temperature * loss_std
        raw_weight = torch.sigmoid(
            (self.spl_threshold - per_sample) / scaled_temp
        )

        # 权重地板：即使最难的样本也保留 min_weight × 1.0 的贡献，
        # 避免"尾部样本梯度彻底归零 → 排序能力退化 → AUC 下降"。
        min_w = self.config.spl_min_weight
        weight = min_w + (1.0 - min_w) * raw_weight

        # --- 7) 归一化加权损失 ---
        weighted = (per_sample * weight).sum() / (weight.sum() + 1e-8)
        loss = weighted + dc_term

        return loss, weight.mean().item()

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train the model with Self-Paced Learning.

        :param train_data: Time series data used for training.
        """
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        self.model = CATCHModel(self.config)
        self.model.to(self.device)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        train_steps = len(self.train_data_loader)
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]

        self.optimizer = torch.optim.Adam(main_params, lr=self.config.lr)
        self.optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=self.config.Mlr)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )

        schedulerM = lr_scheduler.OneCycleLR(
            optimizer=self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.Mlr,
        )

        time_now = time.time()

        for epoch in range(self.config.num_epochs):
            self.spl_epoch = epoch  # 更新当前epoch
            iter_count = 0
            train_loss = []
            avg_weights = []  # 记录自步学习平均权重

            epoch_time = time.time()
            self.model.train()

            step = min(int(len(self.train_data_loader) / 10), 100)
            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                self.optimizerM.zero_grad()

                input = input.float().to(self.device)

                output, output_complex, dcloss = self.model(input)
                output = output[:, :, :]

                # 计算每个样本的重建损失（保留维度用于自步学习）
                rec_loss = self.criterion(output, input)
                # 辅助损失计算
                norm_input = self.model.revin_layer(input, 'transform')
                auxi_loss = self.auxi_loss(output_complex, norm_input)

                # 自步学习损失计算
                loss, avg_weight = self._spl_compute_loss(rec_loss, auxi_loss, dcloss)
                train_loss.append(loss.item())
                avg_weights.append(avg_weight)

                if (i + 1) % step == 0:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()

                if (i + 1) % 100 == 0:
                    avg_loss = np.mean(train_loss[-100:])
                    avg_w = np.mean(avg_weights[-100:])
                    print(
                        "\titers: {0}, epoch: {1} | training loss: {2:.7f} | avg sample weight: {3:.4f} | spl threshold: {4:.7f}".format(
                            i + 1, epoch + 1, avg_loss, avg_w,
                            self.spl_threshold.item() if self.spl_threshold is not None else 0.0
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            # 打印epoch级别的自步学习信息
            avg_epoch_loss = np.average(train_loss)
            avg_epoch_weight = np.average(avg_weights) if avg_weights else 1.0
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            valid_loss = self.detect_validate(self.valid_data_loader, nn.MSELoss(reduction='none'))
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Avg Sample Weight: {3:.4f} | Vali Loss: {4:.7f} | SPL Threshold: {5:.7f}".format(
                    epoch + 1, train_steps, avg_epoch_loss, avg_epoch_weight, valid_loss,
                    self.spl_threshold.item() if self.spl_threshold is not None else 0.0
                )
            )

            # --- SPL 熔断：warmup 末期保存基线；SPL 启动后若 vali 严重恶化就回退 ---
            if (self.config.enable_spl
                    and not self.spl_disabled
                    and epoch == self.config.spl_start_epoch - 1):
                # warmup 刚结束，保存基线
                self.spl_warmup_vali = valid_loss
                self.spl_warmup_state = copy.deepcopy(self.model.state_dict())
                print(f"[SPL] Warmup baseline saved. Vali={valid_loss:.6f}")

            if (self.config.enable_spl
                    and not self.spl_disabled
                    and epoch >= self.config.spl_start_epoch
                    and self.spl_warmup_vali is not None
                    and (not np.isfinite(valid_loss)
                         or valid_loss > self.config.spl_blowup_ratio * self.spl_warmup_vali)):
                # SPL 搞炸了 —— 回退到 warmup 权重，关闭 SPL，继续按 vanilla 训完剩余 epoch
                print(f"[SPL] ABORT: vali={valid_loss:.4f} > {self.config.spl_blowup_ratio}x warmup "
                      f"({self.spl_warmup_vali:.4f}). Reverting to warmup weights; SPL disabled.")
                self.model.load_state_dict(self.spl_warmup_state)
                self.spl_disabled = True
                self.spl_threshold = None
                self.spl_loss_buffer = []
                # 把 warmup 的 vali loss 重新喂给 early_stopping，让它以 warmup 为最佳
                self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
                self.early_stopping(self.spl_warmup_vali, self.model)
            else:
                self.early_stopping(valid_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        attens_energy = []
        test_labels = []
        total_batches = len(self.thre_loader)
        print(f"[detect_score] Inference started: {total_batches} batches")
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                if (i + 1) % 200 == 0 or (i + 1) == total_batches:
                    print(f"[detect_score] thre-energy: {i + 1}/{total_batches}")

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)

        total_train = len(self.train_data_loader)
        total_test = len(self.test_data_loader)
        total_thre = len(self.thre_loader)
        print(f"[detect_label] Inference started: train={total_train} | test={total_test} | thre={total_thre}")

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)

                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)

                if (i + 1) % 200 == 0 or (i + 1) == total_train:
                    print(f"[detect_label] train-energy: {i + 1}/{total_train}")

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                if (i + 1) % 200 == 0 or (i + 1) == total_test:
                    print(f"[detect_label] test-energy: {i + 1}/{total_test}")

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                if (i + 1) % 200 == 0 or (i + 1) == total_thre:
                    print(f"[detect_label] thre-energy: {i + 1}/{total_thre}")

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy