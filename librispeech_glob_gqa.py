import os
import json

FINETUNE_DATA_PATH = "./playground/data/llava_v1_5_mix665k.json"
GQA_DATA_PATH = "./playground/data/coco_plain.json"

with open(FINETUNE_DATA_PATH, 'r', encoding='utf-8') as file:
    data = json.load(file)

gqa = []
for item in data:
    if 'image' in item:
        img_folder = item['image']
        if 'coco' in img_folder:
            gqa.append(item)

with open(GQA_DATA_PATH, 'w', encoding='utf-8') as file:
    json.dump(gqa, file, ensure_ascii=False, indent=4)

print("Finished Processing")


