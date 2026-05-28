#!/bin/bash
mkdir -p log

# Source: 脚本_label.sh command 1 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "PSM.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 1e-05, "auxi_lambda": 0.1, "batch_size": 128, "cf_dim": 32, "d_ff": 16, "d_model": 16, "dc_lambda": 0.1, "e_layers": 1, "head_dim": 32, "lr": 0.005, "n_heads": 16, "num_epochs": 3, "patch_size": 16, "patch_stride": 8, "seq_len": 192, "anomaly_ratio": 3.0, "spl_min_weight": 0.1, "spl_cooldown_epochs": 0}' \
    --gpus 1 --num-workers 1 --timeout 60000 \
    --save-path "label/CATCH_spl_final" \
    > log/label_psm_catch_final.log 2>&1 &
