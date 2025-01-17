import os
import glob
import json
import random


TRAIN_CLEAN_360 = "/zhaowei/data/LibriSpeech/train-clean-360"
trans = glob.glob(os.path.join(TRAIN_CLEAN_360, "**/*.txt"), recursive=True)


# 0. 设定语音识别的Instruction
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

# 1. 构造所有数据
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

# 2. 写入json文件
filename = "playground/data/librispeech_asr_train_clean_360.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print("data is ok")
