#!/bin/bash
mkdir -p log

# Source: 脚本_label.sh command 2 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 1e-05, "auxi_lambda": 0.1, "batch_size": 256, "cf_dim": 32, "d_ff": 128, "d_model": 128, "dc_lambda": 0.2, "e_layers": 2, "head_dim": 128, "lr": 0.0001, "n_heads": 4, "num_epochs": 1, "patch_size": 16, "patch_stride": 8, "score_lambda": 0.5, "seq_len": 192, "anomaly_ratio": 5.0, "spl_min_weight": 0.8, "spl_target_quantile": 0.99, "spl_start_epoch": 0, "spl_cooldown_epochs": 0}' \
    --gpus 2 --num-workers 1 --timeout 60000 \
    --save-path "label/CATCH_spl_final" \
    > log/label_smd_catch_final.log 2>&1 &
