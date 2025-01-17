import os
import argparse
import json
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the text for speech instructions."
    )
    parser.add_argument(
        "--ur5_data_path",
        type=str,
        help="Path to the ur5 dataset directory.",
        default="/storage/zhaowei/data/berkeley_our_ur5",
    )
    parser.add_argument("--debug", type=bool, help="Debug mode.", default=False)
    args = parser.parse_args()

    ur5_data = Path(args.ur5_data_path)

    lang_info = ur5_data / "training" / "auto_lang_ann.npy"
    assert lang_info.exists(), "Invalid data path"
    ann_data = np.load(lang_info, allow_pickle=True).item()
    ann_lang = ann_data["language"]["ann"]
    instruct_set = set(ann_lang)

    instruct2id = [(instruct, idx) for idx, instruct in enumerate(instruct_set)]
    instruct_file = Path(__file__).parent / (ur5_data.stem + ".jsonl")
    with open(instruct_file, "w") as f:
        for item in instruct2id:
            f.write(json.dumps(item) + "\n")

    print("Instructions have been successfully prepared.")

