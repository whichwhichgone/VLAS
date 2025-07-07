"""
Calvin Dataset Text Instruction Extraction

This script extracts and prepares text instructions from the original Calvin robot
manipulation dataset. The Calvin dataset contains robot manipulation tasks with natural
language annotations describing the actions.

Main functionality:
1. Loads language annotations from the Calvin dataset
2. Extracts all unique text instructions from the annotation data
3. Creates an instruction-to-ID mapping for indexing
4. Saves the processed instructions to a JSONL file

The output file serves as input for TTS tools to generate corresponding speech
instructions, enabling the creation of audio-visual robot manipulation datasets
for multimodal learning.

Input: Calvin dataset with language annotations (auto_lang_ann.npy)
Output: JSONL file containing instruction-ID pairs for TTS processing
"""

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
        "--calvin_data_path",
        type=str,
        help="Path to the calvin dataset directory.",
        default="/wangdonglin/calvin/task_ABCD_D",
    )
    parser.add_argument("--debug", type=bool, help="Debug mode.", default=False)
    args = parser.parse_args()

    calvin_data = Path(args.calvin_data_path)

    lang_info = calvin_data / "training" / "lang_annotations" / "auto_lang_ann.npy"
    assert lang_info.exists(), "Invalid data path"
    ann_data = np.load(lang_info, allow_pickle=True).item()
    ann_lang = ann_data["language"]["ann"]
    instruct_set = sorted(set(ann_lang))

    instruct2id = [(instruct, idx) for idx, instruct in enumerate(instruct_set)]
    instruct_file = Path(__file__).parent / (calvin_data.stem + ".jsonl")
    with open(instruct_file, "w") as f:
        for item in instruct2id:
            f.write(json.dumps(item) + "\n")

    print("Instructions have been successfully prepared.")
