"""
Stage 2: Automatic Speech Recognition (ASR) SFT Data Preparation

This script prepares ASR training data for Stage 2 of the VLAS training pipeline.
Stage 2 focuses on building the VLAS-Base model, a VLM model that supports speech
modality input while preserving original performance on vision and language tasks.

The script processes the LibriSpeech train-clean-360 dataset, extracting audio file paths
and their corresponding transcriptions. It transforms this data into JSON format suitable
for supervised fine-tuning (SFT), where each example contains:
- Audio file path
- Conversation pairs with:
    - Human prompt with varied instructions and audio placeholder
    - GPT response with the correct transcription

Key features:
- Random selection from multiple transcription instruction templates
- Random positioning of audio placeholder (before or after instruction)
- Enhanced training diversity for robust speech recognition capabilities

The output JSON file is used for training the model with varied prompt instructions,
improving the model's ability to understand different ways of requesting transcription.
"""

import os
import glob
import json
import random
from tqdm import tqdm

# Path to your own LibriSpeech train-clean-360 dataset
TRAIN_CLEAN_360 = "/zhaowei/data_usr/LibriSpeech/train-clean-360"
trans = glob.glob(os.path.join(TRAIN_CLEAN_360, "**/*.txt"), recursive=True)

# Define varied instruction prompts for speech recognition tasks
prompts = [
    "Transcribe the following speech into text.",
    "Convert the spoken words into written text.",
    "Transform the speech into a written transcript.",
    "Transpose the spoken language into written script.",
    "Translate the spoken words into written text.",
    "Convert the spoken utterances into written transcription.",
    "Transcribe the oral words into text.",
    "Encode the spoken statements into written language.",
    "Produce a text document from the oral speech.",
    "Write out the verbal speech as text.",
]

# Process transcription files to create training samples with prompt variations
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

            # Format data sample for SFT with random instruction and audio placeholder
            instruction = random.sample(prompts, 1)[0]
            if random.random() < 0.5:
                audio_dict = {
                    "id": audio_key,
                    "audio": audio_path,
                    "conversations": [
                        {"from": "human", "value": f"<audio>\n{instruction}"},
                        {"from": "gpt", "value": audio_txt},
                    ],
                }
            else:
                audio_dict = {
                    "id": audio_key,
                    "audio": audio_path,
                    "conversations": [
                        {"from": "human", "value": f"{instruction}\n<audio>"},
                        {"from": "gpt", "value": audio_txt},
                    ],
                }
            data.append(audio_dict)

# Save training data to JSON file
filename = "playground/data/stage2_asr_speech.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print(f"Training data prepared and saved to {filename}")
