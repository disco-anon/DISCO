#!/bin/bash
mkdir -p log

# Source: 脚本_vae_lstm.sh command 12 (self_impl.VAE_LSTM.VAE_LSTM_disco)
nohup "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "PSM.csv" \
    --model-name "self_impl.VAE_LSTM.VAE_LSTM_disco" \
    --model-hyper-params '{"seq_len":100,"batch_size":128,"num_epochs":10,"lr":0.001,"hidden_dim":128,"latent_dim":16,"num_layers":1,"dropout":0.0,"beta":1.0,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"score_source":"last_mse","score_window_agg":"last","score_window_blend":1.0,"score_smooth_window":1,"score_smooth_blend":0.0,"anomaly_ratio":[5,10,15,20,25,28,30],"enable_spl":true,"spl_start_epoch":2,"spl_cooldown_epochs":1,"spl_min_weight":0.6,"spl_init_weight":0.5,"spl_target_quantile":0.9,"spl_temperature":1.0,"spl_gamma":0.9,"spl_blowup_ratio":2.0,"spl_buffer_size":2048,"spl_mode":"easy_first","spl_difficulty_source":"mse"}' \
    --gpus 1 --num-workers 1 --timeout 60000 \
    --save-path "label/VAE_LSTM_disco" > log/label_psm_vae_lstm_disco.log 2>&1 &
