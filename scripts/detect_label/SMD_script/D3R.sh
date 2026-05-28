#!/bin/bash
mkdir -p log

# Source: 脚本_d3r.sh command 4 (self_impl.D3R)
nohup "$PYTHON_BIN" -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "self_impl.D3R" \
    --model-hyper-params '{"window_size":64,"batch_size":8,"num_epochs":8,"patience":3,"lr":0.0001,"weight_decay":0.0001,"train_val_ratio":0.8,"period":1440,"model_dim":512,"ff_dim":2048,"atten_dim":64,"block_num":2,"head_num":8,"dropout":0.6,"time_steps":1000,"beta_start":0.0001,"beta_end":0.02,"t":500,"p":10.0,"d":30,"score_normalize":true,"anomaly_ratio":[0.1,0.5,1,2,3,5,10,15,20,25]}' \
    --gpus 3 --num-workers 1 --timeout 60000 --aggregate_type max \
    --save-path "label/D3R" > log/label_smd_d3r.log 2>&1 &
