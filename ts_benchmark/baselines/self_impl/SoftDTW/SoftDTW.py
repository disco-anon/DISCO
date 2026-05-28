import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_SOFT_DTW_HYPER_PARAMS = {
    "seq_len": 64,
    "gamma": 1.0,
    "train_stride": 1,
    "score_stride": 1,
    "max_train_windows": 2048,
    "aggregation": "mean",
    "normalize_scores": True,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class SoftDTWConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_SOFT_DTW_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _softmin3(a, b, c, gamma):
    if gamma <= 0:
        return min(a, b, c)
    values = np.array([-a / gamma, -b / gamma, -c / gamma], dtype=np.float64)
    max_value = np.max(values)
    return -gamma * (np.log(np.exp(values - max_value).sum()) + max_value)


def _soft_dtw_distance(x, y, gamma):
    """Pure NumPy Soft-DTW value for two arrays shaped (seq_len, channels)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, n = x.shape[0], y.shape[0]
    r = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    r[0, 0] = 0.0

    for i in range(1, m + 1):
        xi = x[i - 1]
        for j in range(1, n + 1):
            diff = xi - y[j - 1]
            cost = float(np.dot(diff, diff))
            r[i, j] = cost + _softmin3(
                r[i - 1, j], r[i, j - 1], r[i - 1, j - 1], gamma
            )
    return r[m, n]


class SoftDTW:
    """
    CATCH-framework adapter for a Soft-DTW prototype-distance anomaly baseline.

    The upstream soft-dtw project is a distance/loss library rather than a full
    anomaly detector. This adapter turns it into a comparable baseline by fitting
    one normal prototype window from the training split and scoring each window
    by its Soft-DTW distance to that prototype.
    """

    def __init__(self, **kwargs):
        self.config = SoftDTWConfig(**kwargs)
        self.model_name = "SoftDTW"
        self.scaler = StandardScaler()
        self.prototype_ = None
        self.train_scores_ = None
        self.train_score_median_ = None
        self.train_score_iqr_ = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def _scaled_df(self, data):
        return pd.DataFrame(
            self.scaler.transform(data.values),
            columns=data.columns,
            index=data.index,
        )

    def _window_starts(self, n_rows, stride):
        seq_len = self.config.seq_len
        if n_rows < seq_len:
            return np.array([], dtype=np.int64)
        return np.arange(0, n_rows - seq_len + 1, max(1, int(stride)), dtype=np.int64)

    def _make_windows(self, values, stride, max_windows=None):
        starts = self._window_starts(values.shape[0], stride)
        if max_windows is not None and len(starts) > max_windows:
            idx = np.linspace(0, len(starts) - 1, int(max_windows), dtype=np.int64)
            starts = starts[idx]
        return np.stack([values[s:s + self.config.seq_len] for s in starts], axis=0)

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
        return windows.mean(axis=0)

    def _score_windows(self, values, stride):
        starts = self._window_starts(values.shape[0], stride)
        scores = np.empty(len(starts), dtype=np.float64)
        for idx, start in enumerate(starts):
            window = values[start:start + self.config.seq_len]
            scores[idx] = _soft_dtw_distance(window, self.prototype_, self.config.gamma)
        return starts, scores

    def _window_scores_to_timestep(self, n_rows, starts, scores):
        if n_rows == 0:
            return np.array([], dtype=np.float64)
        if len(starts) == 0:
            return np.zeros(n_rows, dtype=np.float64)

        seq_len = self.config.seq_len
        out = np.zeros(n_rows, dtype=np.float64)
        counts = np.zeros(n_rows, dtype=np.float64)

        for start, score in zip(starts, scores):
            end = min(n_rows, start + seq_len)
            if self.config.aggregation == "max":
                out[start:end] = np.maximum(out[start:end], score)
                counts[start:end] = 1.0
            else:
                out[start:end] += score
                counts[start:end] += 1.0

        missing = counts == 0
        counts[missing] = 1.0
        out = out / counts
        if missing.any():
            out[missing] = out[~missing][-1] if (~missing).any() else 0.0
        return out

    def _normalize_scores(self, scores):
        if not self.config.normalize_scores:
            return scores
        if self.train_score_median_ is None or self.train_score_iqr_ is None:
            return scores
        return (scores - self.train_score_median_) / (self.train_score_iqr_ + 1e-9)

    def _score_frame(self, data, normalize=True):
        values = data.values.astype(np.float64)
        starts, win_scores = self._score_windows(values, self.config.score_stride)
        scores = self._window_scores_to_timestep(len(data), starts, win_scores)
        if normalize:
            scores = self._normalize_scores(scores)
        return scores

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        self.scaler.fit(train_data.values)
        train_df = self._scaled_df(train_data)
        self.prototype_ = self._fit_prototype(train_df.values.astype(np.float64))
        raw_train_scores = self._score_frame(train_df, normalize=False)
        self.train_score_median_ = np.median(raw_train_scores)
        self.train_score_iqr_ = (
            np.percentile(raw_train_scores, 75)
            - np.percentile(raw_train_scores, 25)
        )
        self.train_scores_ = self._normalize_scores(raw_train_scores)

    def detect_score(self, test: pd.DataFrame):
        if self.prototype_ is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test_df = self._scaled_df(test)
        scores = self._score_frame(test_df)
        return scores, scores

    def detect_label(self, test: pd.DataFrame):
        if self.prototype_ is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test_df = self._scaled_df(test)
        test_scores = self._score_frame(test_df)
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
