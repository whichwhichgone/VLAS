# Preprocess the tfrecords format data from Robot Learning Dataset (RLDS).
import tensorflow_datasets as tfds
import numpy as np
import os
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation as R
import math
from PIL import Image


berkeley_ur5 = "/wangdonglin/open_x_embodiment_datasets/berkeley_autolab_ur5/0.1.0"
dataset = tfds.builder_from_directory(berkeley_ur5).as_dataset()
save_dir = "/storage/zhaowei/data/berkeley_autolab_ur5/training"


global_index = 0
info_indx_list = []
language_ann_list = []

for cnt, episode in tqdm(enumerate(dataset["train"]), dynamic_ncols=True):
    steps = episode["steps"]
    steps = [step for step in steps]

    gripper_prev = 0.0
    for step_id, step in enumerate(steps):
        global_index += 1
        if step_id == 0:
            start_index = global_index
        elif step_id == len(steps) - 1:
            end_index = global_index
        else:
            pass

        '''
        xyz = step["action"]["world_vector"].numpy()
        rpy = step["action"]["rotation_delta"].numpy()
        delta_gripper_closed = step["action"]["gripper_closedness_action"].numpy()
        if delta_gripper_closed == 0.0:
            gripper_state = gripper_prev
        elif delta_gripper_closed == 1.0:
            assert gripper_prev == 0.0
            gripper_state = 1.0 - gripper_prev
            gripper_prev = 1.0
        elif delta_gripper_closed == -1.0:
            assert gripper_prev == 1.0
            gripper_state = 1.0 - gripper_prev
            gripper_prev = 0.0
        else:
            raise ValueError("Invalid gripper action")
        '''

        # action = np.concatenate((xyz, rpy, np.array([gripper_state], dtype=np.float32)), axis=0)
        img = step["observation"]["image"].numpy()
        img_hand = step["observation"]["hand_image"].numpy()
        robot_obs = step["observation"]["robot_state"].numpy()
        lang = step["observation"]["natural_language_instruction"].numpy()
        img_hand = Image.fromarray(img_hand)
        img_hand = img_hand.transpose(Image.ROTATE_180)
        img_hand = np.array(img_hand)
        img_hand = img_hand[..., ::-1]

        state_xyz = robot_obs[6:9]
        state_xyzw = robot_obs[9:13]
        state_gripper = robot_obs[13:14]
        rotation = R.from_quat(state_xyzw)
        state_rpy = rotation.as_euler("xyz", degrees=False)
        if state_rpy[0] > 0:
            state_rpy[0] = state_rpy[0] - math.pi
        else:
            state_rpy[0] = state_rpy[0] + math.pi

        if state_rpy[2] > 0:
            state_rpy[2] = state_rpy[2] - math.pi
        else:
            state_rpy[2] = state_rpy[2] + math.pi
        
        if (step_id + 1) < len(steps):
            next_robot_obs = steps[step_id + 1]["observation"]["robot_state"].numpy()
            next_xyz = next_robot_obs[6:9]
            next_xyzw = next_robot_obs[9:13]
            next_gripper = next_robot_obs[13:14]
            rotation = R.from_quat(next_xyzw)
            next_rpy = rotation.as_euler("xyz", degrees=False)
            if next_rpy[0] > 0:
                next_rpy[0] = next_rpy[0] - math.pi
            else:
                next_rpy[0] = next_rpy[0] + math.pi
            
            if next_rpy[2] > 0:
                next_rpy[2] = next_rpy[2] - math.pi
            else:
                next_rpy[2] = next_rpy[2] + math.pi

            delta_xyz = next_xyz - state_xyz
            delta_rpy = next_rpy - state_rpy
            delta_gripper = next_gripper
            action = np.concatenate((delta_xyz, delta_rpy, delta_gripper), axis=0)
        elif (step_id + 1) == len(steps):
            action = np.zeros((7,))

        index_name = str(global_index).zfill(7) + ".npz"
        index_path = os.path.join(save_dir, index_name)
        np.savez(index_path, rel_actions=action, rgb_static=img, rgb_gripper=img_hand, robot_obs=robot_obs)
    
    info_indx_list.append((start_index, end_index))
    language_ann_list.append(lang.decode("utf-8"))

# Save the meta data
assert len(info_indx_list) == len(language_ann_list)
auto_lang_ann = {"info": {"indx": info_indx_list}, "language": {"ann": language_ann_list}}
np.save(os.path.join(save_dir, "auto_lang_ann.npy"), auto_lang_ann, allow_pickle=True)
