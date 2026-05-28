#!/bin/bash
mkdir -p log

# Source: 脚本_omni_anomaly.sh command 6 (self_impl.OmniAnomaly)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "MSL.csv" \
    --model-name "self_impl.OmniAnomaly" \
    --model-hyper-params '{"seq_len":100,"batch_size":50,"num_epochs":10,"lr":0.001,"hidden_dim":500,"dense_dim":500,"z_dim":3,"beta":1.0,"posterior_flow_type":"nf","nf_layers":20,"std_epsilon":0.0001,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"anomaly_ratio":5.0}' \
    --gpus 0 --num-workers 1 --timeout 60000 \
    --save-path "score/OmniAnomaly" > log/score_msl_omni_anomaly.log 2>&1 &
