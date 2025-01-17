import os
import json

FINETUNE_DATA_PATH = "./playground/data/llava_sft_update.json"
GQA_DATA_PATH = "./playground/data/dummy.json"

with open(FINETUNE_DATA_PATH, 'r', encoding='utf-8') as file:
    data = json.load(file)

gqa = []
cnt = 0
for item in data:
    gqa.append(item)
    cnt += 1
    if cnt > 100:
        break

with open(GQA_DATA_PATH, 'w', encoding='utf-8') as file:
    json.dump(gqa, file, ensure_ascii=False, indent=4)

print("Finished Processing")