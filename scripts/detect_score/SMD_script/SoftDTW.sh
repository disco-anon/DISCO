#!/bin/bash
mkdir -p log

# Source: 脚本_soft_dtw.sh command 9 (self_impl.SoftDTW)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "self_impl.SoftDTW" \
    --model-hyper-params '{"seq_len":24,"gamma":1.0,"train_stride":32,"score_stride":32,"max_train_windows":2048,"anomaly_ratio":5.0}' \
    --gpus 3 \
    --num-workers 1 --timeout 60000 \
    --save-path "score/SoftDTW" \
    > log/score_smd_softdtw.log 2>&1 &
