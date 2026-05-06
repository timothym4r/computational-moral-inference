#!/usr/bin/env bash
set -euo pipefail
source /Users/owner/miniconda3/bin/activate ml

# -------------------------------
# Data Preprocessing Args
# -------------------------------

LOG_FILE="test_folder/moral_word_prediction_test.log"
RUN_DIR="test_folder/recon_typed_run"
INJECTION_SIGNAL_TYPE="recon"

DATA_PREPROCESSING_ARGS=$(cat <<EOF
{
  "source_data_path": "../../data-processing/sample_data/small_sampled_moral_word_prediction_data.json",
  "output_dir": "$RUN_DIR",
  "threshold": 0,
  "model_name": "bert-base-uncased",
  "pooling_method": "mean",
  "reprocess": false,
  "sentence_mask_type": "all_sentence",
  "add_type_tokens": true,
  "store_history_embeddings": true,
  "max_history_per_type": 200
}
EOF
)

# -------------------------------
# MLM Training Args
# -------------------------------
MFD_TRAINING_ARGS=$(cat <<EOF
{
  "input_dir": "$RUN_DIR",
  "output_dir": "$RUN_DIR",
  "model_name": "bert-base-uncased",
  "use_vae": false,
  "use_one_hot": false,
  "latent_dim": 20,
  "alpha": 1.0,
  "beta": 5.0,
  "num_epochs": 1,
  "batch_size": 32,
  "lr_ae": 2e-5,
  "lr_bert": 1e-4,
  "dropout_rate": 0.1,
  "clip_grad_norm": 5.0,
  "weight_decay": 1e-3,
  "scheduler_type": "cosine",
  "early_stopping_patience": 8,
  "train_n_last_layers": 0,
  "inject_embedding": true,
  "pooling_method": "mean",
  "retrain": true,
  "eval_only": false,
  "sentence_mask_type": "all_sentence",
  "threshold": 0,
  "sent_pooler": "mean",
  "decay": 0.9,
  "moral_weight": 1.0,
  "add_type_tokens": true,
  "injection_signal_type": "$INJECTION_SIGNAL_TYPE"
}
EOF
)


python -u main.py \
  --data-processing-args "$DATA_PREPROCESSING_ARGS" \
  --training-args "$MFD_TRAINING_ARGS" \
  > "$LOG_FILE" 2>&1
