#!/bin/bash
mkdir -p log

# Source: 脚本_soft_dtw.sh command 19 (self_impl.SoftDTW_disco)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "self_impl.SoftDTW_disco" \
    --model-hyper-params '{"seq_len":24,"gamma":1.0,"train_stride":32,"score_stride":32,"max_train_windows":2048,"anomaly_ratio":5.0,"num_epochs":5,"num_prototypes":4,"prototype_quantile":0.75,"score_log_every":500,"enable_spl":true,"spl_start_epoch":0,"spl_min_weight":0.3,"spl_init_weight":0.5,"spl_target_quantile":0.95,"spl_temperature":1.0,"spl_gamma":0.9,"spl_cooldown_epochs":0,"spl_buffer_size":2048}' \
    --gpus 3 --num-workers 1 --timeout 60000 \
    --save-path "score/SoftDTW_disco" \
    > log/score_smd_softdtw_disco.log 2>&1 &
