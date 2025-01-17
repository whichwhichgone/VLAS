import os
import json

GPUS_NUM = 8
COCO_PLAIN_PATH = 'playground/data/coco_plain.json'
ASR_PATH = 'playground/data/librispeech_asr_train_clean_360.json'
TXT_PATH = 'playground/data/raw_text_plain.json'
SFT_PATH = 'playground/data/llava_v1_5_mix665k.json'
SFT_UPDATE_PATH = 'playground/data/llava_sft_update.json'


# 1. 获取更新之后的coco对话数据集
merge_data = []
for gpu_id in range(GPUS_NUM):
    json_file = f'playground/data/coco_plain_{gpu_id}_update.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    merge_data += split_data

with open(COCO_PLAIN_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
assert len(merge_data) == len(raw_data)


# 2. 嵌入更新后的coco对话数据集以及librispeech中的语音识别数据集
with open(SFT_PATH, 'r', encoding='utf-8') as f:
    sft_data = json.load(f)

sft_no_coco_and_txt = []
for sft_item in sft_data:
    if 'image' in sft_item:
        img_folder = sft_item['image']
        if 'coco' in img_folder:
            continue
    else:
        continue
    sft_no_coco_and_txt.append(sft_item)

with open(ASR_PATH, 'r', encoding='utf-8') as f:
    asr_data = json.load(f)

with open(TXT_PATH, 'r', encoding='utf-8') as f:
    txt_data = json.load(f)
            
# 3. 将更新后的SFT数据集保存在本地
sft_update = sft_no_coco_and_txt + merge_data + asr_data + txt_data
with open(SFT_UPDATE_PATH, 'w', encoding='utf-8') as f:
    json.dump(sft_update, f, ensure_ascii=False, indent=4)
print('Finshed the data preparing.')



