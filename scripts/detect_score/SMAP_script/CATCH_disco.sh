#!/bin/bash
mkdir -p log

# Source: 脚本_score.sh command 3 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 0.01, "auxi_lambda": 1, "batch_size": 128, "cf_dim": 16, "d_ff": 32, "d_model": 64, "dc_lambda": 1, "dropout": 0.4, "e_layers": 3, "head_dim": 64, "inference_patch_size": 4, "lr": 0.005, "n_heads": 4, "num_epochs": 10, "patch_size": 16, "patch_stride": 8, "score_lambda": 1e-06, "seq_len": 192, "spl_min_weight": 0.6}' \
    --gpus 3 --num-workers 1 --timeout 60000 \
    --save-path "score/CATCH_spl_final" \
    > log/score_smap_catch_final.log 2>&1 &
