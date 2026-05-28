#!/bin/bash
mkdir -p log

# Source: 脚本_score.sh command 1 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "PSM.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 0.001, "auxi_lambda": 0.01, "batch_size": 128, "cf_dim": 16, "d_ff": 32, "d_model": 16, "dc_lambda": 0.05, "dropout": 0.3, "e_layers": 1, "head_dim": 32, "inference_patch_size": 96, "lr": 0.005, "n_heads": 4, "num_epochs": 13, "patch_size": 16, "patch_stride": 8, "patience": 5, "score_lambda": 0.5, "seq_len": 192, "spl_min_weight": 0.95, "spl_target_quantile": 0.99, "spl_cooldown_epochs": 1}' \
    --gpus 6 --num-workers 1 --timeout 60000 \
    --save-path "score/CATCH_spl_final" \
    > log/score_psm_catch_final.log 2>&1 &
