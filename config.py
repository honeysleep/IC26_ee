# config.py

import os
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
MODEL_ID = "facebook/hubert-large-ls960-ft"

# data
EEG_n_chan = 61                     # number of EEG channels

duration = 30
EEG_INPUT_LENGTH = 500 * duration      
SPEECH_LENGTH = 16000 * duration   
TARGET_LENGTH = SPEECH_LENGTH   

# Domain adaptation
DA_NUM_EPOCHS = 100
DA_BATCH_SIZE = 32
DA_LR = 1e-4

# Trainer
TRAIN_ARGS = dict(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    bf16=True,
    group_by_length=True,
    eval_strategy="steps",
    save_steps=500,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
)

# Data
AUDIO_DATA_DIR = "./wav"
EEG_CSV_DIR = "./eeg"
NUM_SUB = 42
