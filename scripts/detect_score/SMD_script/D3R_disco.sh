#!/bin/bash
mkdir -p log

# Source: 脚本_d3r.sh command 19 (self_impl.D3R_disco)
nohup "$PYTHON_BIN" -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "self_impl.D3R_disco" \
    --model-hyper-params '{"window_size":64,"batch_size":8,"num_epochs":8,"patience":3,"lr":0.0001,"weight_decay":0.0001,"train_val_ratio":0.8,"period":1440,"model_dim":512,"ff_dim":2048,"atten_dim":64,"block_num":2,"head_num":8,"dropout":0.6,"time_steps":1000,"beta_start":0.0001,"beta_end":0.02,"t":500,"p":10.0,"d":30,"score_normalize":true,"score_stride":1,"disco_score_fusion":"rank_blend","disco_score_blend":0.7,"disco_score_mode":"fused","train_vanilla_shadow":true,"label_emit_variants":true,"score_emit_variants":true,"anomaly_ratio":5.0,"enable_spl":true,"spl_start_epoch":4,"spl_cooldown_epochs":2,"spl_min_weight":0.85,"spl_init_weight":0.5,"spl_target_quantile":0.85,"spl_temperature":2.0,"spl_gamma":0.9,"spl_blowup_ratio":1.3,"spl_buffer_size":4096,"spl_difficulty_source":"loss"}' \
    --gpus 7 --num-workers 1 --timeout 60000 --aggregate_type max \
    --save-path "score/D3R_disco" > log/score_smd_d3r_disco.log 2>&1 &

# Source: 脚本_d3r_disco.sh command 9 (self_impl.D3R_disco)
nohup /root/miniconda3/envs/tsrl/bin/python -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "self_impl.D3R_disco" \
    --model-hyper-params '{"window_size":64,"batch_size":8,"num_epochs":8,"patience":3,"lr":0.0001,"weight_decay":0.0001,"train_val_ratio":0.8,"period":1440,"model_dim":512,"ff_dim":2048,"atten_dim":64,"block_num":2,"head_num":8,"dropout":0.6,"time_steps":1000,"beta_start":0.0001,"beta_end":0.02,"t":500,"p":10.0,"d":30,"score_normalize":true,"score_stride":1,"disco_score_fusion":"rank_blend","disco_score_blend":0.7,"disco_score_mode":"fused","train_vanilla_shadow":true,"label_emit_variants":true,"score_emit_variants":true,"anomaly_ratio":5.0,"enable_spl":true,"spl_start_epoch":4,"spl_cooldown_epochs":2,"spl_min_weight":0.85,"spl_init_weight":0.5,"spl_target_quantile":0.85,"spl_temperature":2.0,"spl_gamma":0.9,"spl_blowup_ratio":1.3,"spl_buffer_size":4096,"spl_difficulty_source":"loss"}' \
    --gpus 7 --num-workers 1 --timeout 60000 --aggregate_type max \
    --save-path "score/D3R_disco" > log/score_smd_d3r_disco.log 2>&1 &
