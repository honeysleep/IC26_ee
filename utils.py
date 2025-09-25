# utils.py

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pad_sequences(sequences, maxlen=None, dtype="float32", padding="post", truncating="post", value=0.0):
    if not maxlen:
        maxlen = max(len(x) for x in sequences)
    padded_sequences = []
    
    for s in sequences:
        if len(s) > maxlen:
            s = s[:maxlen] if truncating == "pre" else s[-maxlen:]
        else:
            pad_length = maxlen - len(s)
            if padding == "pre":
                s = [value] * pad_length + s
            else:
                s = s + [value] * pad_length
        padded_sequences.append(s)
        
    return np.array(padded_sequences, dtype=dtype)
