import os
import argparse
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


def get_updated_training_data(data_raw, instructions, ur5_audio):
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

            folders = os.listdir(ur5_audio)
            folders = [folder for folder in folders if folder.isdigit()]      
            random_folder = random.choice(folders)
            audio_name = random_folder + "/" + f"{audio_id:04}" + ".wav"

            # Upate the data
            item["conversations"][0]["value"] = updated_instruct
            item["audio"] = audio_name

    return data_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the final training data for the ur5 dataset."
    )
    parser.add_argument(
        "--ur5_json",
        type=str,
        help="ur5 json file without audio data.",
        default="playground/berkeley_ur5/berkeley_autolab_ur5_training.json",
    )
    parser.add_argument(
        "--ur5_audio_json",
        type=str,
        help="ur5 json file with audio data.",
        default="playground/berkeley_ur5_audio/berkeley_autolab_ur5_training_audio.json",
    )
    parser.add_argument(
        "--ur5_audio",
        type=str,
        help="ur5 audio folder.",
        default="/storage/zhaowei/data/ur5_audio/berkeley_autolab_ur5",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        help="ur5 instructions in the dataset.",
        default="playground/berkeley_ur5_audio/berkeley_autolab_ur5.jsonl",
    )
    args = parser.parse_args()

    random.seed(123)
    ur5_json = Path(args.ur5_json)
    ur5_audio_json = Path(args.ur5_audio_json)
    ur5_audio = Path(args.ur5_audio)

    assert (
        ur5_json.stem.split("_training")[0] == ur5_audio.stem
    ), "Check the path."
    assert (
        ur5_json.stem.split("_training")[0]
        == ur5_audio_json.stem.split("_training")[0]
    ), "Check the path."

    with open(ur5_json, "r") as f:
        data_raw = json.load(f)
    
    with open(args.instructions, "r") as f:
        instructions = [json.loads(line) for line in f]

    data_updated = get_updated_training_data(data_raw, instructions, ur5_audio)
    with open(ur5_audio_json, "w") as f:
        json.dump(data_updated, f, indent=4)
    print("ur5 audio training data has been prepared.")
