#!/bin/bash
mkdir -p log

# Source: 脚本_score.sh command 2 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"Mlr": 1e-05, "batch_size": 64, "cf_dim": 64, "d_ff": 128, "d_model": 128, "e_layers": 3, "head_dim": 32, "lr": 0.0001, "n_heads": 16, "num_epochs": 5, "patch_size": 8, "patch_stride": 8, "seq_len": 192, "spl_min_weight": 0.6}' \
    --gpus 7 --num-workers 1 --timeout 60000 \
    --save-path "score/CATCH_spl_final" \
    > log/score_smd_catch_final.log 2>&1 &
