# data.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, Audio
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from transformers import AutoProcessor


class CSVDataset(Dataset):
    def __init__(self, folder_path: str, target_length: int):
        self.data = []
        self.target_length = target_length

    def to_hf_dataset(self) -> HFDataset:
        data_dict = {"input_values": [d.numpy().tolist() for d in self.data]}
        return HFDataset.from_dict(data_dict)

    def __len__(self):
        return len(self.data) if isinstance(self.data, torch.Tensor) else 0

    def __getitem__(self, idx):
        if len(self.data) == 0:
            raise IndexError("Dataset is empty")
        return self.data[idx]

def uppercase(example):
    return {"text": example["text"].upper()}

def prepare_dataset(batch, processor: AutoProcessor, max_input_length: int):
    audio = batch["audio"]
    batch = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["text"],
        padding=True,
        max_length=max_input_length,
        truncation=True,
    )
    batch["input_length"] = len(batch["input_values"][0])
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
