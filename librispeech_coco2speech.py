import json
import random
import numpy as np
import time
import soundfile
import glob
import kaldiio
import os
from espnet2.bin.tts_inference import Text2Speech
from tqdm import tqdm
import argparse
import shortuuid

import torch


SFT_PATH = 'playground/data/llava_v1_5_mix665k.json'
ASR_PATH = 'playground/data/librispeech_asr_train_clean_100.json'
TRIPLE_MODALITY_PATH = '/zhaowei/data/LLaVA-Audio/'
TTS_MODEL_PATH = "/zhaowei/models/kan-bayashi_libritts_xvector_vits/exp/tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth"
SFT_UPDATE_PATH = 'playground/data/llava_sft_update.json'

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def tts_espnet(model, xvectors, text, save_path, index):
    spks = list(xvectors.keys())

    # randomly select speaker
    random_spk_idx = np.random.randint(0, len(spks))
    spk = spks[random_spk_idx]
    spembs = xvectors[spk]
    spk_id = spk.split('_')[0]

    with torch.no_grad():
        wav = text2speech(text, spembs=spembs)["wav"]
    
    os.makedirs(os.path.join(save_path, spk_id), exist_ok=True)
    audio_path = os.path.join(save_path, spk_id, index + '.wav')
    soundfile.write(audio_path, wav.cpu().numpy(), text2speech.fs, "PCM_16")
    audio_target = spk_id + '/' + index + '.wav'
    return audio_target, spk_id


def prepare_text_for_tts(raw_text: str):
    """
        Here, the format of raw_text may be:
        1. Please provide a short description for this region: [0.04, 0.57, 0.63, 0.71].
        2. Please provide the bounding box coordinate of the region this sentence describes: brown car with sticker of star in rear window.
        3. Where is he most likely pushing the things to?\nA. forest\nB. temple\nC. grocery store\nD. airport taxi 
        4. What color is the stop sign?
    """
    
    if "Please provide the bounding box coordinate of the region this sentence describes:" in raw_text:
        target_text = raw_text.split(":", maxsplit=1)[-1].strip()
        raw_text_upd = raw_text.split(":", maxsplit=1)[0].strip() + "\n<audio>"
    elif "Please provide a short description for this region:" in raw_text:
        target_text = raw_text.split(":", maxsplit=1)[0].strip()
        raw_text_upd = "<audio>\n" + raw_text.split(":", maxsplit=1)[-1].strip()
    # for multiple-choice questions
    elif "\n" in raw_text:
        target_text = raw_text.split("\n", maxsplit=1)[0].strip()
        raw_text_upd = "<audio>\n" + raw_text.split("\n", maxsplit=1)[-1].strip()
    elif "Answer the question using a single word or phrase" in raw_text:
        print(raw_text)
        raise ValueError("Wrong data format.")
    else:
        target_text = raw_text
        raw_text_upd = "<audio>\n"
    
    return target_text, raw_text_upd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=None)
    args = parser.parse_args()
    if args.json_file is None or args.chunk_idx is None:
        raise ValueError("Wrong input data.")

    # 0. 加载原始的针对coco数据集的训练样本、初始化TTS实例以及说话人x-vector合集
    set_seed()
    with open(args.json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    text2speech = Text2Speech.from_pretrained(
            model_file=TTS_MODEL_PATH,
            device="cuda",
            # Only for Tacotron 2 & Transformer
            threshold=0.5,
            # Only for Tacotron 2
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3,
            # Only for FastSpeech & FastSpeech2 & VITS
            speed_control_alpha=1.0,
            # Only for VITS
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )

    model_dir = os.path.dirname(TTS_MODEL_PATH)
    if text2speech.use_spembs:
        xvector_ark = [p for p in glob.glob(f"{model_dir}/../../dump/**/spk_xvector.ark", recursive=True) if "train-clean-460" in p][0]
        xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}


    # 1. 在原始训练样本中，随机抽取from人类的文本问题，并将其转为音频
    debug_txt = []
    for raw_idx, raw_item in tqdm(enumerate(raw_data), desc='Saving the audios:'):
        idx_group = []
        for conv_idx, conv in enumerate(raw_item["conversations"]):
            if conv['from'] == 'human':
                idx_group.append(conv_idx)

        # 默认从带图像的下次会话进行采样
        if idx_group[1:] == []:
            continue

        sample_idx = random.sample(idx_group[1:], 1)[0]

        # 确定将采样到的目标文本中哪些部分用音频表示
        target_text, raw_text_upd = prepare_text_for_tts(raw_item["conversations"][sample_idx]['value'])
        target_idx = raw_item["id"] + "_" + shortuuid.uuid()[:6]

        # 调用接口合成音频，保存音频，并返回保存的音频地址
        target_audio, spk_id = tts_espnet(text2speech, xvectors, target_text, TRIPLE_MODALITY_PATH, target_idx)

        # 更新原始数据
        raw_item["conversations"][sample_idx]['value'] = raw_text_upd
        raw_item['audio'] = target_audio
        raw_item['speaker'] = spk_id
        debug_txt.append(target_text)

    # 2. 保留当前update过后的各个split信息
    split_file_update = f'playground/data/coco_plain_{args.chunk_idx}_update.json'
    with open(split_file_update, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)

    # 3. 保存用于debug的语音合成的文本数据
    debug_file = f'playground/data/coco_plain_{args.chunk_idx}_debug.txt'
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(debug_txt))
