# main.py

import torch
import copy
import numpy as np
from torch.utils.data import TensorDataset
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import AutoProcessor, AutoModelForCTC, Trainer, TrainingArguments

from config import *
from utils import seed_everything, pad_sequences
from data import CSVDataset, uppercase, prepare_dataset, DataCollatorCTCWithPadding
from train import train_domain_adaptation, compute_metrics, update_dataset_with_inference


def main():
    seed_everything(SEED)

    # Processor & ASR model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    asr_model = AutoModelForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Load wav dataset
    wav_dataset = load_dataset("audiofolder", data_dir=AUDIO_DATA_DIR, split="train")
    
    all_spoken_datasets = []
    for spoken_id in range(1, NUM_SUB + 1):
        sd = copy.deepcopy(wav_dataset)
        sd = sd.map(lambda x, sid=spoken_id: {"spoken_id": sid})
        all_spoken_datasets.append(sd)
        
    wav_dataset = concatenate_datasets(all_spoken_datasets)
    wav_dataset = wav_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    # Load EEG dataset
    eeg_dataset = CSVDataset(EEG_CSV_DIR, target_length=EEG_INPUT_LENGTH)

    wav_dataset = wav_dataset.map(uppercase)
    wav_dataset = wav_dataset.map(lambda b: prepare_dataset(b, processor, SPEECH_LENGTH))

    source_input_values = wav_dataset["input_values"]  
    source_input_values_padded = pad_sequences(source_input_values, dtype="float32")
    source_input_tensor = torch.FloatTensor(source_input_values_padded)

    target_input_values = eeg_dataset["input_values"]  
    target_input_values_padded = pad_sequences(target_input_values, dtype="float32")
    target_input_tensor = torch.FloatTensor(target_input_values_padded)

    source_labels_tensor = torch.zeros(len(source_input_tensor)).long()
    target_labels_tensor = torch.ones(len(target_input_tensor)).long()

    source_dataset = TensorDataset(source_input_tensor, source_labels_tensor)
    target_dataset = TensorDataset(target_input_tensor, target_labels_tensor)

    # Train domain adaptation
    target_encoder, discriminator = train_domain_adaptation(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        device=DEVICE,
        num_epochs=DA_NUM_EPOCHS,
        batch_size=DA_BATCH_SIZE,
        lr=DA_LR,
    )

    torch.save(target_encoder.state_dict(), "target_encoder.pt")
    torch.save(discriminator.state_dict(), "discriminator.pt")
    
    target_encoder.load_state_dict(torch.load("target_encoder.pt", map_location=DEVICE))
    target_encoder.to(DEVICE).eval()

    wav_dataset = wav_dataset.map(
        lambda batch: update_dataset_with_inference(batch, eeg_dataset, target_encoder, DEVICE),
        batched=True,
        batch_size=32,
        keep_in_memory=True,
        load_from_cache_file=False,
    )

    wav_dataset = wav_dataset.remove_columns(["audio", "text", "spoken_id", "input_length"])
    wav_dataset = wav_dataset.train_test_split(test_size=0.2)

    data_collator = DataCollatorCTCWithPadding(processor=processor, max_length=SPEECH_LENGTH, padding="longest")
    training_args = TrainingArguments(**TRAIN_ARGS)

    trainer = Trainer(
        model=asr_model,
        args=training_args,
        train_dataset=wav_dataset["train"],
        eval_dataset=wav_dataset["test"],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics(processor),
    )

    trainer.train()

if __name__ == "__main__":
    main()
