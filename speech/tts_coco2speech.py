"""
Convert COCO QA text-image samples to speech-image samples

This script's main functionality:
1. Load COCO QA samples containing text-image question-answer pairs
2. Extract text questions from human conversations in the samples
3. Use TTS (Text-to-Speech) model to synthesize audio from the extracted text
4. Save the generated audio files and update the original samples with audio paths
5. Convert text-image QA samples to speech-image QA samples for multimodal training

Input: coco_qa_splits_{chunk_idx}.json - Split COCO QA dataset files
Output: coco_qa_splits_{chunk_idx}_update.json - Updated samples with audio paths
        Generated audio files saved to TRIPLE_MODALITY_PATH
"""

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


# Specify the PATH for saving generated audios, and the PATH for TTS model
TRIPLE_MODALITY_PATH = "/zhaowei/data/LLaVA-Audio-TTS"
TTS_MODEL_PATH = "/zhaowei/models/kan-bayashi_libritts_xvector_vits/exp/tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth"


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
    spk_id = spk.split("_")[0]

    with torch.no_grad():
        wav = text2speech(text, spembs=spembs)["wav"]

    os.makedirs(os.path.join(save_path, spk_id), exist_ok=True)
    audio_path = os.path.join(save_path, spk_id, index + ".wav")
    soundfile.write(audio_path, wav.cpu().numpy(), text2speech.fs, "PCM_16")
    audio_target = spk_id + "/" + index + ".wav"
    return audio_target, spk_id


def prepare_text_for_tts(raw_text: str):
    """
    Here, the format of raw_text may be:
    1. Please provide a short description for this region: [0.04, 0.57, 0.63, 0.71].
    2. Please provide the bounding box coordinate of the region this sentence describes: brown car with sticker of star in rear window.
    3. Where is he most likely pushing the things to?\nA. forest\nB. temple\nC. grocery store\nD. airport taxi
    4. What color is the stop sign?
    """

    if (
        "Please provide the bounding box coordinate of the region this sentence describes:"
        in raw_text
    ):
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

    # 0. Load original text-image samples from coco QA, initialize TTS instance and speaker x-vector collection
    set_seed()
    with open(args.json_file, "r", encoding="utf-8") as f:
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
        xvector_ark = [
            p
            for p in glob.glob(
                f"{model_dir}/../../dump/**/spk_xvector.ark", recursive=True
            )
            if "train-clean-460" in p
        ][0]
        xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}

    # 1. Randomly select text questions from human conversations and convert them to audio
    debug_txt = []
    for raw_idx, raw_item in tqdm(enumerate(raw_data), desc="Saving the audios:"):
        idx_group = []
        for conv_idx, conv in enumerate(raw_item["conversations"]):
            if conv["from"] == "human":
                idx_group.append(conv_idx)

        # Sample from turns after the first, since we want to reserve the first turn which has visual inputs
        if idx_group[1:] == []:
            continue
        sample_idx = random.sample(idx_group[1:], 1)[0]

        # Determine which parts of the sampled target text will be represented by audio
        target_text, raw_text_upd = prepare_text_for_tts(
            raw_item["conversations"][sample_idx]["value"]
        )
        target_idx = raw_item["id"] + "_" + shortuuid.uuid()[:6]

        # Call the TTS interface to synthesize audio, save it, and return the saved audio path
        target_audio, spk_id = tts_espnet(
            text2speech, xvectors, target_text, TRIPLE_MODALITY_PATH, target_idx
        )
        raw_item["conversations"][sample_idx]["value"] = raw_text_upd
        raw_item["audio"] = target_audio
        raw_item["speaker"] = spk_id
        debug_txt.append(target_text)

    # 2. Save the updated split information
    split_file_update = f"playground/data/coco_qa_splits_{args.chunk_idx}_update.json"
    with open(split_file_update, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)

    # 3. Save the textual instructions used for speech synthesis for debugging
    debug_file = f"playground/data/coco_qa_splits_{args.chunk_idx}_debug.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(debug_txt))
