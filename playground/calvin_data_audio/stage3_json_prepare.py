"""
This script converts pure text instruction JSON training data into mixed-modal training 
data that includes both text and audio instructions. It creates a balanced dataset where 
approximately 60% of the training samples use audio instructions instead of text.

Key functionalities:
1. Loads existing Calvin JSON training data with text instructions
2. Randomly selects 60% of samples to replace text instructions with audio
3. Maps text instructions to corresponding synthesized audio files
4. Updates conversation format to include audio placeholders
5. Maintains consistent audio placeholder positioning after image tokens

Audio integration process:
- Parses conversation structure to extract text instructions
- Maps instructions to audio IDs using instruction-to-ID mapping
- Randomly selects speaker voice from available synthesized audio folders
- Replaces text instruction with "<audio>" placeholder token
- Links appropriate audio file path to the training sample

The audio placeholder "<audio>" is consistently positioned immediately after the 
image placeholder "<image>" in the conversation flow, ensuring standardized 
multimodal input sequence for model training.

Input: Calvin JSON with text instructions + instruction-ID mapping + audio files
Output: Enhanced JSON with mixed text/audio instructions for multimodal VLA training
"""

import os
import argparse
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


def get_updated_training_data(data_raw, instructions, calvin_audio):
    instruct2id = {instruct[0]: instruct[1] for instruct in instructions}

    for item in tqdm(data_raw):
        instruct = item["conversations"][0]
        instruct_parts = instruct["value"].split("\n")
        assert instruct_parts[1] in instruct2id, "Check the json data or instructions."

        random_number = random.random()
        if random_number > 0.4:
            audio_id = instruct2id[instruct_parts[1]]
            instruct_parts[1] = "<audio>"
            updated_instruct = "\n".join(instruct_parts)

            folders = os.listdir(calvin_audio)
            random_folder = random.choice(folders)
            audio_name = random_folder + "/" + f"{audio_id:04}" + ".wav"

            # Upate the data
            item["conversations"][0]["value"] = updated_instruct
            item["audio"] = audio_name

    return data_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the final training data for the calvin dataset."
    )
    parser.add_argument(
        "--calvin_json",
        type=str,
        help="Calvin json file without audio data.",
        default="./playground/calvin_data/task_ABCD_D_training_r5.json",
    )
    parser.add_argument(
        "--calvin_audio_json",
        type=str,
        help="Calvin json file with audio data.",
        default="./playground/calvin_data_audio/task_ABCD_D_training_r5_audio.json",
    )
    parser.add_argument(
        "--calvin_audio",
        type=str,
        help="Calvin audio folder.",
        default="/zhaowei/data/calvin_audio/task_ABCD_D",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        help="Calvin instructions in the dataset.",
        default="./playground/calvin_data_audio/task_ABCD_D.jsonl",
    )
    args = parser.parse_args()

    random.seed(123)
    calvin_json = Path(args.calvin_json)
    calvin_audio_json = Path(args.calvin_audio_json)
    calvin_audio = Path(args.calvin_audio)

    assert (
        calvin_json.stem.split("_training")[0] == calvin_audio.stem
    ), "Check the path."
    assert (
        calvin_json.stem.split("_training")[0]
        == calvin_audio_json.stem.split("_training")[0]
    ), "Check the path."

    with open(calvin_json, "r") as f:
        data_raw = json.load(f)

    with open(args.instructions, "r") as f:
        instructions = [json.loads(line) for line in f]

    data_updated = get_updated_training_data(data_raw, instructions, calvin_audio)
    with open(calvin_audio_json, "w") as f:
        json.dump(data_updated, f, indent=4)
    print("Calvin audio training data has been prepared.")
