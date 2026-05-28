#!/bin/bash
mkdir -p log

# Source: 脚本_baseline.sh command 1 (merlion.AutoEncoder)
nohup python -u ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "MSL.csv" --model-name "merlion.AutoEncoder" --model-hyper-params '{}' --gpus 5 --num-workers 1 --timeout 60000 --save-path "label/AutoEncoder" > log/label_msl_autoencoder.log 2>&1 & echo "[MSL-AE] PID=$!"
