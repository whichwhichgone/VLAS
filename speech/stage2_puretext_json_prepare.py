"""
This script extracts text-only entries from the Llava dataset by filtering out
all items containing images. The filtered data is saved to a new JSON file
for use in pure text-based training scenarios.
"""

import os
import json
from tqdm import tqdm


ORIGINAL_SFT_DATA_PATH = "./playground/data/llava_v1_5_mix665k.json"
PURE_TEXT_DATA_PATH = "./playground/data/stage2_pure_text.json"

with open(ORIGINAL_SFT_DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

items = []
for item in tqdm(data):
    if "image" not in item:
        items.append(item)

with open(PURE_TEXT_DATA_PATH, "w", encoding="utf-8") as file:
    json.dump(items, file, ensure_ascii=False, indent=4)

print(f"Finished Processing, pure text items saved to {PURE_TEXT_DATA_PATH}.")
