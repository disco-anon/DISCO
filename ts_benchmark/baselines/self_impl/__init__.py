__all__ = [
    "VAR_model",
    "LOF",
    "DCdetector",
    "AnomalyTransformer",
    "ModernTCN",
    "DualTF",
    "TFAD",
    "SensitiveHUE",
    "SensitiveHUE_disco",
    "SoftDTW",
    "SoftDTW_disco",
    "OmniAnomaly",
    "OmniAnomaly_disco",
    "InterFusion",
    "InterFusion_disco",
    "D3R",
    "D3R_disco",
    "VAE_LSTM",
    "VAE_LSTM_disco",
]


from ts_benchmark.baselines.self_impl.LOF.lof import LOF

try:
    from ts_benchmark.baselines.self_impl.VAR.VAR import VAR_model
except ModuleNotFoundError:
    VAR_model = None

from ts_benchmark.baselines.self_impl.DCdetector.DCdetector import DCdetector
from ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer import AnomalyTransformer
from ts_benchmark.baselines.self_impl.ModernTCN.ModernTCN import ModernTCN
from ts_benchmark.baselines.self_impl.DualTF.DualTF import DualTF
from ts_benchmark.baselines.self_impl.TFAD.TFAD import TFAD
from ts_benchmark.baselines.self_impl.SensitiveHUE.SensitiveHUE import SensitiveHUE
from ts_benchmark.baselines.self_impl.SensitiveHUE_disco.SensitiveHUE_disco import SensitiveHUE_disco
from ts_benchmark.baselines.self_impl.SoftDTW.SoftDTW import SoftDTW
from ts_benchmark.baselines.self_impl.SoftDTW.SoftDTW_disco import SoftDTW_disco
from ts_benchmark.baselines.self_impl.OmniAnomaly.OmniAnomaly import OmniAnomaly, OmniAnomaly_disco
from ts_benchmark.baselines.self_impl.InterFusion.InterFusion import InterFusion, InterFusion_disco
from ts_benchmark.baselines.self_impl.D3R.D3R import D3R, D3R_disco
from ts_benchmark.baselines.self_impl.VAE_LSTM.VAE_LSTM import VAE_LSTM, VAE_LSTM_disco
