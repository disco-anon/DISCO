#!/bin/bash
mkdir -p log

# Source: 脚本_label.sh command 4 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SWAT.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 1e-05, "batch_size": 64, "cf_dim": 64, "d_ff": 128, "d_model": 128, "e_layers": 3, "head_dim": 64, "lr": 0.0001, "n_heads": 16, "num_epochs": 3, "patch_size": 8, "patch_stride": 8, "seq_len": 192, "anomaly_ratio": 3.0, "spl_min_weight": 0.5, "spl_target_quantile": 0.9, "spl_start_epoch": 0, "spl_cooldown_epochs": 0, "spl_blowup_ratio": 3.0}' \
    --gpus 4 --num-workers 1 --timeout 60000 \
    --save-path "label/CATCH_spl_final" \
    > log/label_swat_catch_final.log 2>&1 &
