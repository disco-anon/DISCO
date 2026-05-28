#!/bin/bash
mkdir -p log

# Source: 脚本_vae_lstm.sh command 9 (self_impl.VAE_LSTM.VAE_LSTM)
nohup "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_score_multi_config.json" \
    --data-name-list "SMD.csv" \
    --model-name "self_impl.VAE_LSTM.VAE_LSTM" \
    --model-hyper-params '{"seq_len":100,"batch_size":128,"num_epochs":10,"lr":0.001,"hidden_dim":128,"latent_dim":16,"num_layers":1,"dropout":0.0,"beta":1.0,"patience":5,"train_val_ratio":0.7,"gradient_clip_norm":10.0,"score_normalize":true,"score_source":"last_mse","score_window_agg":"last","score_window_blend":1.0,"score_smooth_window":1,"score_smooth_blend":0.0,"anomaly_ratio":5.0}' \
    --gpus 0 --num-workers 1 --timeout 60000 \
    --save-path "score/VAE_LSTM" > log/score_smd_vae_lstm.log 2>&1 &
