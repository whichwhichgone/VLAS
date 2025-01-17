import os
import argparse
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


def get_updated_training_data(data_raw, instructions, calvin_audio):
    instruct2id = {instruct[0] : instruct[1] for instruct in instructions}

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
        default="./playground/calvin_data/task_ABCD_D_training_r20.json",
    )
    parser.add_argument(
        "--calvin_audio_json",
        type=str,
        help="Calvin json file with audio data.",
        default="./playground/calvin_data_audio/task_ABCD_D_training_r20_audio.json",
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
