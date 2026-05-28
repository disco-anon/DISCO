#!/bin/bash
mkdir -p log

# Source: 脚本_soft_dtw.sh command 7 (self_impl.SoftDTW)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "PSM.csv" \
    --model-name "self_impl.SoftDTW" \
    --model-hyper-params '{"seq_len":24,"gamma":1.0,"train_stride":16,"score_stride":16,"max_train_windows":2048,"anomaly_ratio":5.0}' \
    --gpus 1 \
    --num-workers 1 --timeout 60000 \
    --save-path "score/SoftDTW" \
    > log/score_psm_softdtw.log 2>&1 &
