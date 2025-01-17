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


TARGET_IMG_SIZE = 334
random.seed(1234)

def get_llm_data(
    instruction: str,
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


def process_episide(episode: tuple, data_path: Path, split: str, future_k: int = 5, hist_k: int = None):
    llm_data_list = []
    ann, index_range = episode[0], episode[1]
    if hist_k is not None:
        robot_obs_queue = [np.zeros((15,), dtype=np.float64)] * hist_k

    for step in tqdm(range(index_range[0], index_range[1] + 1)):
        next_actions = []
        step_data = str(step).zfill(7) + ".npz"
        step_data = data_path / split / step_data
        assert step_data.exists(), "Invalid data path"

        for delta in range(future_k):
            future_step = step + delta
            future_data = str(future_step).zfill(7) + ".npz"
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
        if hist_k is not None:
            robot_obs_queue.append(robot_obs)
            robot_obs_queue.pop(0)

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
            Path("/storage/zhaowei/data") / data_path.stem / "vla_processed" / split,
            exist_ok=True,
        )
        img_concat.save(
            Path("/storage/zhaowei/data")
            / data_path.stem
            / "vla_processed"
            / split
            / sample
        )

        if hist_k is not None:
            robot_obs = np.concatenate(robot_obs_queue, axis=0)
        llm_item = get_llm_data(ann, split, sample, next_actions, robot_obs)
        llm_data_list.append(llm_item)
    return llm_data_list


def build_json_lang(data_path, debug):
    data_path = Path(data_path)
    for split in ["training"]:
        lang_info = data_path / split / "auto_lang_ann.npy"
        assert lang_info.exists(), "Invalid data path"
        ann_data = np.load(lang_info, allow_pickle=True).item()
        lang_ann = ann_data["language"]["ann"]
        lang_index = ann_data["info"]["indx"]
        partial_episode_process = partial(
            process_episide, data_path=data_path, split=split, hist_k=None
        )

        if not debug:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(
                   partial_episode_process, zip(lang_ann, lang_index)
                )
            llm_data_list = [item for sub_results in results for item in sub_results]
        else:
            for zip_item in zip(lang_ann, lang_index):
                results = partial_episode_process(zip_item)
            llm_data_list = results

        target_file = Path(__file__).parent / (data_path.stem + "_" + split + ".json")
        with open(target_file, "w") as json_file:
            json.dump(llm_data_list, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load the ur5 data and convert it into a json file."
    )
    parser.add_argument(
        "--ur5_data_path",
        type=str,
        help="Path to the ur5 dataset directory.",
        default="/storage/zhaowei/data/berkeley_autolab_ur5",
    )
    parser.add_argument("--debug", type=bool, help="Debug mode.", default=False)
    args = parser.parse_args()
    build_json_lang(args.ur5_data_path, args.debug)
