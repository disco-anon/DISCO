#!/bin/bash
mkdir -p log

# Source: 脚本_sensitivehue.sh command 15 (self_impl.SensitiveHUE)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "self_impl.SensitiveHUE" \
    --model-hyper-params '{"seq_len":24,"batch_size":256,"num_epochs":20,"lr":0.001,"dim_model":128,"head_num":4,"dim_hidden_fc":256,"encode_layer_num":1,"alpha":1.0,"patience":10}' \
    --gpus 1 --num-workers 1 --timeout 60000 \
    --save-path "score/SensitiveHUE" > log/score_smap_sensitivehue.log 2>&1 &
