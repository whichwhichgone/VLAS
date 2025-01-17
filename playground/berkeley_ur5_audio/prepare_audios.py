import json
import random
import numpy as np
import time
import math
import soundfile
import glob
import kaldiio
import os
from espnet2.bin.tts_inference import Text2Speech
from tqdm import tqdm
import argparse
import shortuuid
from pathlib import Path

import torch


TTS_MODEL_PATH = "/zhaowei/models/kan-bayashi_libritts_xvector_vits/exp/tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def synthesize_calvin_speech(
    instructions, task_name, calvin_audio_dir, text2speech, xvectors
):
    # Only use the first 500 speakers for simplicity
    xvectors_500 = {}
    for xvector_key, xvector_value in xvectors.items():
        if len(xvectors_500) >= 500:
            break
        if xvector_key.split("_")[0] not in xvectors_500:
            xvectors_500[xvector_key.split("_")[0]] = xvector_value

    for instruction in tqdm(instructions):
        for xvector_key, xvector_value in xvectors_500.items():
            with torch.inference_mode():
                wav = text2speech(instruction[0], spembs=xvector_value)["wav"]
            spk_id = xvector_key
            os.makedirs(
                os.path.join(calvin_audio_dir, task_name, spk_id), exist_ok=True
            )
            audio_path = os.path.join(
                calvin_audio_dir, task_name, spk_id, f"{instruction[1]:04}.wav"
            )
            soundfile.write(audio_path, wav.cpu().numpy(), text2speech.fs, "PCM_16")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the ur5 json data to support for speech instructions."
    )
    parser.add_argument(
        "--instructions",
        type=str,
        help="Path to the ur5 dataset instructions.",
        default="playground/berkeley_ur5_audio/berkeley_autolab_ur5.jsonl",
    )
    parser.add_argument(
        "--ur5_audio_dir",
        type=str,
        help="Path to the calvin audio directory.",
        default="/storage/zhaowei/data/ur5_audio",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=0,
    )
    parser.add_argument("--debug", type=bool, help="Debug mode.", default=False)
    args = parser.parse_args()

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

    task_name = os.path.splitext(os.path.basename(args.instructions))[0]
    with open(args.instructions, "r") as f:
        instructions = [json.loads(line) for line in f]
    instructions = get_chunk(instructions, args.num_chunks, args.chunk_idx)

    # Synthesize the speech for instructions
    synthesize_calvin_speech(
        instructions, task_name, args.ur5_audio_dir, text2speech, xvectors
    )
    print(f"Chunk {args.chunk_idx} successfully synthesized speech for instructions.")
