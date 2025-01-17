import os
import json

COCO_PLAIN_PATH = 'playground/data/coco_plain.json'
GPUS_NUM = 8

with open(COCO_PLAIN_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

splits = []
for idx in range(0, len(raw_data), len(raw_data) // GPUS_NUM):
    splits.append(raw_data[idx: idx + len(raw_data) // GPUS_NUM])

if len(splits) == GPUS_NUM + 1:
    split_extra  = splits.pop()
    splits[-1] += split_extra

for split_idx, split_item in enumerate(splits):
    split_file = f'playground/data/coco_plain_{split_idx}.json'
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_item, f, ensure_ascii=False, indent=4) 

print("finished")