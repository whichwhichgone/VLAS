"""
Stage 1: Alignment Data Preparation

This script prepares training data for Stage 1 of the VLAS training pipeline.
Stage 1 focuses on aligning speech and text modalities by creating paired examples.

The script processes the LibriSpeech train-clean-100 dataset, extracting audio file 
paths and their corresponding transcriptions. It transforms this data into JSON format 
suitable for supervised fine-tuning (SFT), where each example contains:
- Audio file path
- Conversation pairs with:
    - Human prompt containing audio placeholder
    - GPT response with the correct transcription

The output JSON file is used for training the model to transcribe speech to text, 
establishing foundational audio-text alignment capabilities.
"""

import os
import glob
import json
from tqdm import tqdm

# Path to your own LibriSpeech train-clean-100 dataset
TRAIN_CLEAN_100 = "/zhaowei/data_usr/LibriSpeech/train-clean-100"
trans = glob.glob(os.path.join(TRAIN_CLEAN_100, "**/*.txt"), recursive=True)

# Process transcription files to extract audio-text pairs
data = []
for tran in tqdm(trans, desc="Processing transcription files", ncols=100):
    with open(tran, "r") as f:
        for line in f:
            splits = line.split(" ", maxsplit=1)
            audio_key = splits[0].strip()
            audio_txt = splits[1].strip()
            audio_name = audio_key + ".flac"
            audio_path = os.path.join(
                audio_key.split("-")[0],
                audio_key.split("-")[1],
                audio_name,
            )

            # format the data sample for SFT
            audio_dict = {
                "id": audio_key,
                "audio": audio_path,
                "conversations": [
                    {"from": "human", "value": "<audio>\n"},
                    {"from": "gpt", "value": audio_txt},
                ],
            }
            data.append(audio_dict)

# Save to LLaVA-Pretrain compatible directory structure
filename = "playground/data/LLaVA-Pretrain/stage1_speech_data.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Stage 1 json data prepared and saved to {filename}")
