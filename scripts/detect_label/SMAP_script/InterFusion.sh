#!/bin/bash
mkdir -p log

# Source: 脚本_interfusion.sh command 5 (self_impl.InterFusion)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "self_impl.InterFusion" \
    --model-hyper-params '{"seq_len":100,"batch_size":100,"num_epochs":20,"pretrain_epochs":5,"lr":0.001,"hidden_dim":256,"z_dim":3,"z2_dim":13,"dropout":0.0,"beta":1.0,"pretrain_beta":1.0,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"anomaly_ratio":[5,10,13,15,20]}' \
    --gpus 4 --num-workers 1 --timeout 60000 \
    --save-path "label/InterFusion" > log/label_smap_interfusion.log 2>&1 &
