# DiSCo: Difficulty-Guided Sample Coordination for Reconstruction-Based Time Series Anomaly Detection

This repository provides an implementation of **DiSCo**, a difficulty-guided coordinated training framework for unsupervised time series anomaly detection (TSAD).

DiSCo addresses the training bias caused by anomaly-contaminated training data by using reconstruction error as a unified difficulty signal to coordinate data utilization, forward information flow, and gradient updates during training. The method is model-agnostic and can be applied to various reconstruction-based TSAD backbones.

---

## Repository Structure

```text
DISCO/
|-- config/                 # Benchmark configuration files for different evaluation settings
|-- dataset/                # Dataset directory; place the downloaded datasets here
|-- scripts/                # Shell scripts for reproducing experiments
|   |-- detect_label/       # Scripts for label-based anomaly detection evaluation
|   `-- detect_score/       # Scripts for score-based anomaly detection evaluation
|-- ts_benchmark/           # Core benchmark and model implementation
|   |-- baselines/          # Baseline TSAD models and DiSCo-enhanced variants
|   |-- common/             # Shared constants and common utilities
|   |-- data/               # Data loading and preprocessing utilities
|   |-- evaluation/         # Evaluation strategies and metric computation
|   |-- models/             # Model factory and model interfaces
|   |-- report/             # Result aggregation and reporting utilities
|   `-- utils/              # General helper utilities
|-- README.md               # Project documentation
`-- requirements.txt        # Python dependencies
```

## Requirements

The code is implemented in Python. Required dependencies can be installed via:

pip install -r requirements.txt


## Datasets

Prepare Data. You can obtained the well pre-processed datasets from [OneDrive](https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc) or [BaiduCloud](https://pan.baidu.com/s/1W7UoAWKZjoukSZ74FTipYA?pwd=2255). (This may take some time, please wait patiently.) Then place the downloaded data under the folder `./dataset`. 


## Running Experiments

We provide the experiment scripts for baselines and baselines + disco under the folder `./scripts`. For example you can reproduce a experiment result as the following:

```shell
sh ./scripts/detect_label/MSL_script/Intufision_disco.sh

sh ./scripts/detect_score/MSL_script/Intufision_disco.sh
```



