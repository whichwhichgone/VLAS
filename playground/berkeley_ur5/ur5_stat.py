import numpy as np
import os
from tqdm import tqdm


def load_npz_files(folder_path):
    """
    Load all .npy files from a given folder path.
    :param folder_path: The directory where the .npy files are stored.
    :return: A list of NumPy arrays.
    """
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    robot_obs_arrays = []
    action_arrays = []
    for file_ in tqdm(npz_files):
        array = np.load(os.path.join(folder_path, file_))
        robot_obs = array['robot_obs']
        action = array['rel_actions']

        robot_obs_arrays.append(robot_obs)
        action_arrays.append(action)
    return robot_obs_arrays, action_arrays


def calculate_mean_std(arrays):
    """
    Calculate the mean and variance across multiple NumPy arrays along each dimension.
    :param arrays: List of NumPy arrays with the same shape.
    :return: A tuple of (mean, variance) arrays.
    """
    # Stack arrays along a new axis
    stacked_arrays = np.stack(arrays, axis=0)

    # Calculate the mean and variance along the new axis
    mean = np.mean(stacked_arrays, axis=0)
    std = np.std(stacked_arrays, axis=0)

    return mean, std


def calculate_max_min(arrays):
    stacked_arrays = np.stack(arrays, axis=0)
    max_ = np.max(stacked_arrays, axis=0)
    min_ = np.min(stacked_arrays, axis=0)
    return max_, min_


# Example usage
folder_path = '/storage/zhaowei/data/berkeley_autolab_ur5/training'
robot_obs_arrays, action_arrays = load_npz_files(folder_path)
mean, std = calculate_mean_std(robot_obs_arrays)
max_, min_ = calculate_mean_std(action_arrays)

# Output the results
print("Mean across all robot obs arrays:\n", mean)
print("Std across all robot obs arrays:\n", std)
print("Max across all robot action arrays:\n", max_)
print("Min across all robot action arrays:\n", min_)
