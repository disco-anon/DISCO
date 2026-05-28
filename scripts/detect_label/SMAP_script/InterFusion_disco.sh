#!/bin/bash
mkdir -p log

# Source: 脚本_interfusion_disco.sh command 5 (self_impl.InterFusion_disco)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "self_impl.InterFusion_disco" \
    --model-hyper-params '{"seq_len":100,"batch_size":100,"num_epochs":20,"pretrain_epochs":5,"lr":0.001,"hidden_dim":256,"z_dim":3,"z2_dim":13,"dropout":0.0,"beta":1.0,"pretrain_beta":1.0,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"train_vanilla_shadow":true,"label_emit_variants":true,"score_emit_variants":true,"disco_blend_alpha":0.5,"anomaly_ratio":[5,10,13,15,20],"enable_spl":true,"spl_start_epoch":2,"spl_cooldown_epochs":2,"spl_min_weight":0.7,"spl_target_quantile":0.9,"spl_temperature":0.8,"spl_difficulty_source":"loss"}' \
    --gpus 4 --num-workers 1 --timeout 60000 \
    --save-path "label/InterFusion_disco" > log/label_smap_interfusion_disco.log 2>&1 &
