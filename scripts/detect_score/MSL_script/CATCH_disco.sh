#!/bin/bash
mkdir -p log

# Source: 脚本_score.sh command 5 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "MSL.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 5e-05, "batch_size": 128, "cf_dim": 64, "d_ff": 256, "d_model": 128, "e_layers": 3, "head_dim": 64, "lr": 0.0003, "n_heads": 2, "num_epochs": 10, "patch_size": 16, "patch_stride": 8, "patience": 5, "seq_len": 192, "spl_min_weight": 0.6, "spl_target_quantile": 0.95, "spl_cooldown_epochs": 2}' \
    --gpus 1 --num-workers 1 --timeout 60000 \
    --save-path "score/CATCH_spl_final" \
    > log/score_msl_catch_final.log 2>&1 &
