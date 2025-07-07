"""
Data Merger for VLAS Stage2 Training

This script merges multiple sub-datasets into a single training file for the Stage2 
fine-tuning phase of the VLAS model.

Input Files:
- playground/data/coco_qa_splits_{gpu_id}_update.json: COCO data splits processed by multiple GPUs
- playground/data/coco_qa_plain.json: Original COCO Q&A data for validation
- playground/data/llava_v1_5_mix665k.json: Original LLaVA SFT mixed dataset  
- playground/data/stage2_asr_speech.json: ASR speech data for training
- playground/data/stage2_pure_text.json: Pure text conversation data

Output File:
- playground/data/stage2_vlas_final.json: Final merged dataset for Stage2 training
"""

import os
import json

GPUS_NUM = 8
COCO_PLAIN_PATH = "playground/data/coco_qa_plain.json"
ASR_PATH = "playground/data/stage2_asr_speech.json"
TXT_PATH = "playground/data/stage2_pure_text.json"
SFT_PATH = "playground/data/llava_v1_5_mix665k.json"
SFT_UPDATE_PATH = "playground/data/stage2_vlas_final.json"


# 1. Get the updated coco-based speech-image format dialogue data
merge_data = []
for gpu_id in range(GPUS_NUM):
    json_file = f"playground/data/coco_qa_splits_{gpu_id}_update.json"
    with open(json_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    merge_data += split_data

with open(COCO_PLAIN_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
assert len(merge_data) == len(raw_data)


# 2. Process and merge multiple data sources:
#    - Replace text-image dialogue data with speech-image format
#    - Update pure text data
#    - Incorporate librispeech-based ASR data
with open(SFT_PATH, "r", encoding="utf-8") as f:
    sft_data = json.load(f)

sft_no_coco_and_txt = []
for sft_item in sft_data:
    if "image" in sft_item:
        img_folder = sft_item["image"]
        if "coco" in img_folder:
            continue
    else:
        continue
    sft_no_coco_and_txt.append(sft_item)

with open(ASR_PATH, "r", encoding="utf-8") as f:
    asr_data = json.load(f)

with open(TXT_PATH, "r", encoding="utf-8") as f:
    txt_data = json.load(f)

sft_update = sft_no_coco_and_txt + merge_data + asr_data + txt_data
with open(SFT_UPDATE_PATH, "w", encoding="utf-8") as f:
    json.dump(sft_update, f, ensure_ascii=False, indent=4)

print("Finshed the data preparing.")
