#!/bin/bash
mkdir -p log

# Source: 脚本_label.sh command 3 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 5e-05, "auxi_lambda": 0.01, "batch_size": 128, "cf_dim": 64, "d_ff": 256, "d_model": 128, "dropout": 0.4, "e_layers": 1, "head_dim": 64, "head_dropout": 0.3, "lr": 0.0005, "n_heads": 16, "num_epochs": 5, "patch_size": 16, "patch_stride": 8, "patience": 10, "score_lambda": 1e-06, "seq_len": 192, "anomaly_ratio": 2.0}' \
    --gpus 3 --num-workers 1 --timeout 60000 \
    --save-path "label/CATCH_spl_final" \
    > log/label_smap_catch_final.log 2>&1 &
