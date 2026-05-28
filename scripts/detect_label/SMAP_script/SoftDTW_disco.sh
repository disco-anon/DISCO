#!/bin/bash
mkdir -p log

# Source: 脚本_soft_dtw.sh command 15 (self_impl.SoftDTW_disco)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "self_impl.SoftDTW_disco" \
    --model-hyper-params '{"seq_len":24,"gamma":1.0,"train_stride":16,"score_stride":16,"max_train_windows":2048,"anomaly_ratio":7.0,"num_epochs":5,"num_prototypes":2,"prototype_quantile":0.6,"score_log_every":500,"enable_spl":true,"spl_start_epoch":0,"spl_min_weight":0.5,"spl_init_weight":0.5,"spl_target_quantile":0.9,"spl_temperature":0.75,"spl_gamma":0.9,"spl_cooldown_epochs":0,"spl_buffer_size":2048}' \
    --gpus 4 --num-workers 1 --timeout 60000 \
    --save-path "label/SoftDTW_disco" \
    > log/label_smap_softdtw_disco.log 2>&1 &
