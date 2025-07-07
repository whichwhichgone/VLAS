"""
Extract COCO-based QA samples from original LLaVA SFT data

This script's main functionality:
1. Filter QA samples containing COCO images from the LLaVA v1.5 mixed dataset
2. Save the filtered text-image QA samples to a separate JSON file
3. These QA samples will be used to convert to speech-image QA samples for training multimodal speech-vision-language models

Input: llava_v1_5_mix665k.json - Original LLaVA SFT mixed dataset
Output: coco_plain.json - Filtered COCO-related QA samples
"""

import os
import json
from tqdm import tqdm


ORIGINAL_SFT_DATA_PATH = "./playground/data/llava_v1_5_mix665k.json"
COCO_DATA_PATH = "./playground/data/coco_qa_plain.json"

with open(ORIGINAL_SFT_DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

items = []
for item in tqdm(data):
    if "image" in item:
        img_folder = item["image"]
        if "coco" in img_folder:
            items.append(item)

with open(COCO_DATA_PATH, "w", encoding="utf-8") as file:
    json.dump(items, file, ensure_ascii=False, indent=4)

print(f"Finished Processing, target COCO QA items saved to {COCO_DATA_PATH}.")
