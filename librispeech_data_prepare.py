import os
import glob
import json

TRAIN_CLEAN_100 = "/zhaowei/data/LibriSpeech/train-clean-100"
trans = glob.glob(os.path.join(TRAIN_CLEAN_100, "**/*.txt"), recursive=True)

# 0. 构造所有数据
data = []
for tran in trans:
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

            # 组装单元数据
            audio_dict = {
                "id": audio_key,
                "audio": audio_path,
                "conversations": [
                    {"from": "human", "value": "<audio>\n"},
                    {"from": "gpt", "value": audio_txt},
                ],
            }
            data.append(audio_dict)

# 1. 写入json文件
filename = "playground/data/LLaVA-Pretrain/librispeech_asr_train_clean_100.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print("data is ok")
