"""
Calvin Dataset to JSON Conversion for VLA Training

This script converts the original Calvin robot manipulation dataset into JSON format
suitable for Vision-Language-Action (VLA) model training. The Calvin dataset contains
robot manipulation episodes with visual observations, language instructions, and actions.

Key functionalities:
1. Processes Calvin dataset episodes with multi-view camera observations
2. Combines static and gripper camera views into single composite images
3. Extracts future action sequences for multi-step prediction training
4. Generates training samples with language instructions and robot observations
5. Creates JSON files with file paths referencing separately stored multimodal data

Data processing pipeline:
- Loads language annotations and task information from Calvin dataset
- Processes each episode step to extract visual observations and actions
- Resizes and concatenates static and gripper camera images (336x336 total)
- Saves processed images to local directory with unique identifiers
- Creates JSON entries with image paths, conversations, and action targets

Input: Original Calvin dataset with .npz episode files and language annotations
Output: JSON training files with image paths + separately saved processed images
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image
from functools import partial
import shortuuid
import random


TARGET_IMG_SIZE = 336


def get_llm_data(
    instruction: str,
    task: str,
    split: str,
    sample: str,
    next_actions: list,
    robot_obs: np.array,
):
    flattened_actions = [action.flatten() for action in next_actions]
    flattened_actions = np.hstack(flattened_actions)
    actions_string = " ".join(map(str, flattened_actions))
    flattened_robot_obs = robot_obs.flatten()
    robot_obs_string = " ".join(map(str, flattened_robot_obs))

    llm_item = {
        "id": Path(sample).stem,
        "image": str(Path(split) / sample),
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + instruction + "\n" + robot_obs_string,
            },
            {"from": "gpt", "value": actions_string},
        ],
        "embody": True,
    }

    return llm_item


def process_episide(episode: tuple, data_path: Path, split: str, future_k: int = 5):
    llm_data_list = []
    ann, task, index_range = episode[0], episode[1], episode[2]

    for step in tqdm(range(index_range[0], index_range[1] + 1)):
        next_actions = []
        step_data = "episode_" + str(step).zfill(7) + ".npz"
        step_data = data_path / split / step_data
        assert step_data.exists(), "Invalid data path"

        for delta in range(future_k):
            future_step = step + delta
            future_data = "episode_" + str(future_step).zfill(7) + ".npz"
            future_data = data_path / split / future_data
            if future_step <= index_range[1]:
                assert future_data.exists(), "Invalid data path"
                actions = np.load(future_data)["rel_actions"]
            else:
                break
            next_actions.append(actions)

        if len(next_actions) < future_k:
            pad_num = future_k - len(next_actions)
            pad_action = next_actions[-1]
            next_actions.extend([pad_action] * pad_num)
        assert len(next_actions) == future_k, "Invalid future actions"

        total_data = np.load(step_data)
        rgb_static = total_data["rgb_static"]
        rgb_gripper = total_data["rgb_gripper"]
        robot_obs = total_data["robot_obs"]

        img_static = Image.fromarray(rgb_static)
        img_static = img_static.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_gripper = Image.fromarray(rgb_gripper)
        img_gripper = img_gripper.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        img_concat.paste(img_static, (0, 0))
        img_concat.paste(img_gripper, (0, TARGET_IMG_SIZE // 2))

        uuid = shortuuid.ShortUUID().random(length=7)
        sample = uuid + "_" + str(step).zfill(7) + ".jpg"
        os.makedirs(
            Path("/zhaowei/data/calvin")
            / data_path.stem
            / f"vla_processed_r{future_k}"
            / split,
            exist_ok=True,
        )
        img_concat.save(
            Path("/zhaowei/data/calvin")
            / data_path.stem
            / f"vla_processed_r{future_k}"
            / split
            / sample
        )
        llm_item = get_llm_data(ann, task, split, sample, next_actions, robot_obs)
        llm_data_list.append(llm_item)
    return llm_data_list


def build_json_lang(data_path, debug):
    data_path = Path(data_path)
    for split in ["training", "validation"]:
        lang_info = data_path / split / "lang_annotations" / "auto_lang_ann.npy"
        assert lang_info.exists(), "Invalid data path"
        ann_data = np.load(lang_info, allow_pickle=True).item()
        lang_ann = ann_data["language"]["ann"]
        lang_task = ann_data["language"]["task"]
        lang_index = ann_data["info"]["indx"]
        partial_episode_process = partial(
            process_episide, data_path=data_path, split=split, future_k=5
        )

        if not debug:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(
                    partial_episode_process, zip(lang_ann, lang_task, lang_index)
                )
            llm_data_list = [item for sub_results in results for item in sub_results]
        else:
            for zip_item in zip(lang_ann, lang_task, lang_index):
                results = partial_episode_process(zip_item)

            # debug mode: only process the last zip_item
            llm_data_list = results

        target_file = Path(__file__).parent / (
            data_path.stem + "_" + split + f"_r{future_k}.json"
        )
        with open(target_file, "w") as json_file:
            json.dump(llm_data_list, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load the original calvin data and convert it into a json file."
    )
    parser.add_argument(
        "--calvin_data_path",
        type=str,
        help="Path to the calvin dataset directory.",
        default="/wangdonglin/calvin/task_ABCD_D",
    )
    parser.add_argument("--debug", type=bool, help="Debug mode.", default=False)
    args = parser.parse_args()
    random.seed(1234)
    np.random.seed(1234)
    build_json_lang(args.calvin_data_path, args.debug)
