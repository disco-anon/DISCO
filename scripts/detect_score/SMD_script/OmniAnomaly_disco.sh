#!/bin/bash
mkdir -p log

# Source: 脚本_omni_anomaly.sh command 19 (self_impl.OmniAnomaly_disco)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "self_impl.OmniAnomaly_disco" \
    --model-hyper-params '{"seq_len":100,"batch_size":50,"num_epochs":10,"lr":0.001,"hidden_dim":500,"dense_dim":500,"z_dim":3,"beta":1.0,"posterior_flow_type":"nf","nf_layers":20,"std_epsilon":0.0001,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"anomaly_ratio":5.0,"enable_spl":true,"spl_start_epoch":4,"spl_cooldown_epochs":2,"spl_min_weight":0.95,"spl_target_quantile":0.85,"spl_temperature":1.0,"spl_difficulty_source":"nll"}' \
    --gpus 6 --num-workers 1 --timeout 60000 \
    --save-path "score/OmniAnomaly_disco" > log/score_smd_omni_anomaly_disco.log 2>&1 &
