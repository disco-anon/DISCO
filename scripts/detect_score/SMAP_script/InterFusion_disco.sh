#!/bin/bash
mkdir -p log

# Source: 脚本_interfusion_disco.sh command 10 (self_impl.InterFusion_disco)
nohup python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMAP.csv" \
    --model-name "self_impl.InterFusion_disco" \
    --model-hyper-params '{"seq_len":100,"batch_size":100,"num_epochs":20,"pretrain_epochs":5,"lr":0.001,"hidden_dim":256,"z_dim":3,"z2_dim":13,"dropout":0.0,"beta":1.0,"pretrain_beta":1.0,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"train_vanilla_shadow":false,"label_emit_variants":false,"score_emit_variants":true,"disco_blend_alpha":0.5,"score_source":"last_nll","score_source_variants":["last_nll","mean_nll","max_nll","sum_nll","meanmax_nll","loss"],"score_include_inverse":true,"anomaly_ratio":5.0,"enable_spl":false,"spl_start_epoch":2,"spl_cooldown_epochs":2,"spl_min_weight":0.95,"spl_target_quantile":0.85,"spl_temperature":1.0,"spl_difficulty_source":"nll"}' \
    --gpus 6 --num-workers 1 --timeout 60000 \
    --save-path "score/InterFusion_disco" > log/score_smap_interfusion_disco.log 2>&1 &
