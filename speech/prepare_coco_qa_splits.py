"""
Split COCO QA dataset into multiple sub-files for multi-GPU processing

This script's main functionality:
1. Load the original COCO QA JSON file containing text-image QA samples
2. Split the dataset into multiple sub-files based on the number of GPUs available
3. Each sub-file will be processed by a separate GPU to generate speech-image QA datasets
4. This enables parallel processing of speech generation using multiple GPUs

Input: coco_qa_plain.json - Original COCO QA dataset
Output: coco_qa_plain_{idx}.json - Split QA datasets for each GPU
"""

import os
import json
from tqdm import tqdm


COCO_PLAIN_PATH = "playground/data/coco_qa_plain.json"
GPUS_NUM = 8

with open(COCO_PLAIN_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

splits = []
for idx in range(0, len(raw_data), len(raw_data) // GPUS_NUM):
    splits.append(raw_data[idx : idx + len(raw_data) // GPUS_NUM])

if len(splits) == GPUS_NUM + 1:
    split_extra = splits.pop()
    splits[-1] += split_extra

for split_idx, split_item in tqdm(enumerate(splits)):
    split_file = f"playground/data/coco_qa_splits_{split_idx}.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_item, f, ensure_ascii=False, indent=4)

print("Finished splitting COCO QA dataset into multiple sub-files.")
