#!/bin/bash
mkdir -p log

# Source: 脚本_omni_anomaly.sh command 15 (self_impl.OmniAnomaly_disco)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "self_impl.OmniAnomaly_disco" \
    --model-hyper-params '{"seq_len":100,"batch_size":50,"num_epochs":10,"lr":0.001,"hidden_dim":500,"dense_dim":500,"z_dim":3,"beta":1.0,"posterior_flow_type":"nf","nf_layers":20,"std_epsilon":0.0001,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"anomaly_ratio":[5,10,13,15,20],"enable_spl":true,"spl_start_epoch":2,"spl_cooldown_epochs":1,"spl_min_weight":0.7,"spl_target_quantile":0.9,"spl_temperature":0.8,"spl_difficulty_source":"loss"}' \
    --gpus 4 --num-workers 1 --timeout 60000 \
    --save-path "label/OmniAnomaly_disco" > log/label_smap_omni_anomaly_disco.log 2>&1 &
