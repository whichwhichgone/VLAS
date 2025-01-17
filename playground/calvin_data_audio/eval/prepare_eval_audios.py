import os
from pathlib import Path
import json
import argparse
import yaml
import kaldiio
import glob
import soundfile
import torch
from espnet2.bin.tts_inference import Text2Speech
from tqdm import tqdm


TTS_MODEL_PATH = "/zhaowei/models/kan-bayashi_libritts_xvector_vits/exp/tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth"

def synthesize_eval_speech(instructions, calvin_audio_dir, text2speech, xvectors):

    xvectors_eval = {}
    for xvector_key, xvector_value in xvectors.items():
        if xvector_key.split("_")[0] not in xvectors_eval:
            xvectors_eval[xvector_key.split("_")[0]] = xvector_value

    for instruction in tqdm(instructions.values()):
        for xvector_key, xvector_value in xvectors_eval.items():
            with torch.inference_mode():
                wav = text2speech(instruction[0], spembs=xvector_value)["wav"]
            spk_id = xvector_key
            os.makedirs(
                os.path.join(calvin_audio_dir, spk_id), exist_ok=True
            )
            audio_path = os.path.join(
                calvin_audio_dir, spk_id, f"{instruction[1]:04}.wav"
            )
            soundfile.write(audio_path, wav.cpu().numpy(), text2speech.fs, "PCM_16")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval audios.")
    parser.add_argument(
        "--eval_instructions",
        type=str,
        default="playground/calvin_data_audio/eval/new_playtable_validation.yaml",
    )
    parser.add_argument(
        "--eval_audios", type=str, default="/zhaowei/data/calvin_eval_audio"
    )
    args = parser.parse_args()

    with open(args.eval_instructions, "r") as f:
        eval_instructions = yaml.safe_load(f)

    # Update the eval instructions to have the index info.
    for idx, (item_key, item_value) in enumerate(eval_instructions.items()):
        item_value.append(idx)

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
            if "test-clean" in p
        ][0]
        xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}

    synthesize_eval_speech(
        eval_instructions, args.eval_audios, text2speech, xvectors
    )

    print("Evaluation audios have been successfully prepared.")
