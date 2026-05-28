#!/bin/bash
mkdir -p log

# Source: 脚本_sensitivehue.sh command 20 (self_impl.SensitiveHUE_disco)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "self_impl.SensitiveHUE_disco" \
    --model-hyper-params '{"seq_len":24,"batch_size":256,"num_epochs":20,"lr":0.001,"dim_model":128,"head_num":4,"dim_hidden_fc":256,"encode_layer_num":1,"alpha":1.0,"patience":10,"enable_spl":true,"spl_start_epoch":2,"spl_cooldown_epochs":2,"spl_min_weight":0.3,"spl_init_weight":0.5,"spl_target_quantile":0.95,"spl_temperature":1.0,"spl_gamma":0.9,"spl_blowup_ratio":2.0,"spl_buffer_size":2048}' \
    --gpus 7 --num-workers 1 --timeout 60000 \
    --save-path "score/SensitiveHUE_disco" > log/score_smap_sensitivehue_disco.log 2>&1 &
