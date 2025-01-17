import os
import argparse
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


# how many QA data used for co-training
NUM_QA_SLICES = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the final training data for the ur5 dataset."
    )

    parser.add_argument(
        "--ur5_audio_json",
        type=str,
        help="ur5 json file with audio data.",
        default="playground/our_ur5_audio/berkeley_our_ur5_training_audio.json",
    )
    parser.add_argument(
        "--qa_data",
        type=str,
        help="QA data for co-training.",
        default="playground/data",
    )
    parser.add_argument(
        "--final_data",
        type=str,
        help="final data for co-training.",
        default="playground/our_ur5_audio/ur5_training_audio_cotrain_part.json",
    )
    args = parser.parse_args()

    random.seed(123)
    ur5_audio_json = Path(args.ur5_audio_json)
    qa_data = Path(args.qa_data)
    final_data = Path(args.final_data)

    with open(ur5_audio_json, "r") as f:
        robot_data = json.load(f)

    merge_data = []
    for gpu_id in range(NUM_QA_SLICES):
        json_file = qa_data / f"coco_plain_{gpu_id}_update.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        merge_data += split_data
    
    # Unify the data path
    for item in merge_data:
        assert "image" in item or "audio" in item
        if "image" in item:
            item["image"] = "cotrain_image/" + item["image"]
        if "audio" in item:
            item["audio"] = "cotrain_audio/" + item["audio"]
    
    # Increase the weight for robot data
    robot_data = robot_data * 10

    total_data = merge_data + robot_data
    random.shuffle(total_data)
    with open(final_data, "w") as f:
        json.dump(total_data, f, indent=4)
    print("ur5 final co-training data has been prepared.")