# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import evaluate
from transformers import AutoProcessor, AutoModelForCTC, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets, Audio
import copy

from config import *
from models import TargetEncoder, Discriminator


def train_domain_adaptation(
    source_dataset: TensorDataset,
    target_dataset: TensorDataset,
    device: torch.device,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    target_encoder = TargetEncoder(input_channels=EEG_n_chan, input_length=EEG_INPUT_LENGTH, target_length=TARGET_LENGTH).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.AdamW(target_encoder.parameters(), lr=lr)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for (source_input, _), (target_input, _) in zip(source_loader, target_loader):
            source_input = source_input.to(device)
            target_input = target_input.to(device)

            # G step
            optimizer_g.zero_grad()
            target_output = target_encoder(target_input)  
            source_pred = discriminator(source_input)     
            target_pred = discriminator(target_output)   
            
            g_loss = criterion(target_pred, torch.ones_like(target_pred))
            g_loss.backward()
            optimizer_g.step()

            # D steps
            optimizer_d.zero_grad()
            source_pred = discriminator(source_input.detach())
            target_pred = discriminator(target_output.detach())
            d_source_loss = criterion(source_pred, torch.ones_like(source_pred))
            d_target_loss = criterion(target_pred, torch.zeros_like(target_pred))
            d_loss = d_source_loss + d_target_loss
            d_loss.backward()
            optimizer_d.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")

    return target_encoder, discriminator

def compute_metrics(processor: AutoProcessor):
    def _compute(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        cer = evaluate.load("cer")
        cer_score = cer.compute(predictions=pred_str, references=label_str)
        return {"cer": cer_score}
    
    return _compute

def update_dataset_with_inference(batch, eeg_dataset, target_encoder, device):
    batch_size = len(batch["input_values"])
    eeg_inputs = eeg_dataset["input_values"][:batch_size]
    updated_inputs = []
    
    for sample in eeg_inputs:
        sample_tensor = torch.FloatTensor(sample).to(device)
        if sample_tensor.dim() == 1:
            sample_tensor = sample_tensor.unsqueeze(0)
        if sample_tensor.dim() == 2:
            sample_tensor = sample_tensor.unsqueeze(1)
            
        if sample_tensor.size(1) != EEG_n_chan:
            sample_tensor = sample_tensor.repeat(1, EEG_n_chan, 1)
            
        with torch.no_grad():
            output = target_encoder(sample_tensor)
        updated_inputs.append(output.squeeze().cpu().numpy().tolist())
        
    batch["input_values"] = updated_inputs
    
    return batch
