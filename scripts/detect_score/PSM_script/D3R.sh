#!/bin/bash
mkdir -p log

# Source: 脚本_d3r.sh command 7 (self_impl.D3R)
nohup "$PYTHON_BIN" -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "PSM.csv" \
    --model-name "self_impl.D3R" \
    --model-hyper-params '{"window_size":64,"batch_size":8,"num_epochs":8,"patience":3,"lr":0.0001,"weight_decay":0.0001,"train_val_ratio":0.8,"period":1440,"model_dim":512,"ff_dim":2048,"atten_dim":64,"block_num":2,"head_num":8,"dropout":0.6,"time_steps":1000,"beta_start":0.0001,"beta_end":0.02,"t":500,"p":10.0,"d":30,"score_normalize":true,"anomaly_ratio":5.0}' \
    --gpus 1 --num-workers 1 --timeout 60000 --aggregate_type max \
    --save-path "score/D3R" > log/score_psm_d3r.log 2>&1 &
