#!/bin/bash
mkdir -p log

# Source: 脚本_score.sh command 4 (catch.CATCH)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SWAT.csv" \
    --model-name "catch.CATCH" \
    --model-hyper-params '{"auxi_lambda": 0, "batch_size": 32, "inference_patch_size": 256, "inference_patch_stride": 32, "patch_size": 256, "patch_stride": 64, "score_lambda": 0, "seq_len": 2048, "spl_min_weight": 0.6}' \
    --gpus 0 --num-workers 1 --timeout 60000 \
    --save-path "score/CATCH_spl_final" \
    > log/score_swat_catch_final.log 2>&1 &
