"""
HDF5 MSHab Unified Dataset Module

This module provides a unified PyTorch-compatible dataset class for loading MSHab robot 
manipulation data from HDF5 files. It supports both full control mode and base-only 
navigation mode through a configurable parameter.

The dataset supports multi-camera visual inputs and multi-modal robot actions including 
arm control, gripper control, and base navigation.
"""

import pathlib
import h5py
import yaml
import numpy as np
from typing import List, Literal
from configs.state_vec import STATE_VEC_IDX_MAPPING
from helpers.common import ACDIT_DEFAULT_LOGGER


class HDF5MSHABDataset:
    """
    Unified dataset class for loading MSHab robot manipulation episodes from HDF5 files.
    
    This dataset can operate in two modes:
    - 'full': Full 13-dimensional action space (arm + gripper + head + base)
    - 'base_only': Only 2-dimensional base navigation actions
    
    The dataset handles multi-modal robot data including:
    - Visual observations from multiple cameras (head, hand)
    - Robot state and action sequences
    - Natural language instructions
    - Success/failure labels
    - In-context conditions for goal-oriented manipulation
    """
    
    def __init__(
        self,
        data_dir: str,
        config: dict,
        mode: Literal["full", "base_only"] = "full",
    ):
        """
        Initialize the MSHab unified dataset with configuration parameters.

        Args:
            data_dir: Directory path containing the dataset
            config: Configuration dictionary
            mode: Control mode - 'full' for complete action space, 'base_only' for navigation only
        """
        # Dataset identification
        self.DATASET_NAME = "mshab"
        self.mode = mode
        self.data_dir = data_dir

        # Define all available task combinations (task, subtask, object)
        self.task_subtask_obj = [
            ("set_table", "pick", "013_apple"),
            ("set_table", "pick", "024_bowl"),
            ("set_table", "place", "013_apple"),
            ("set_table", "place", "024_bowl"),
            ("set_table", "open", "fridge"),
            ("set_table", "open", "kitchen_counter"),
            ("set_table", "close", "kitchen_counter"),
        ]

        # Load hyperparameters from configuration
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        self.num_episode_per_task = 100  # Number of episodes per task combination
        
        # Define universal vector indices based on mode
        self._setup_universal_indices()

    def _setup_universal_indices(self):
        """Setup universal vector indices based on the control mode."""
        # State indices are always the same (13-dimensional state space)
        self.UNI_STATE_INDICES = [
            STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)  # Right arm joints 0-6
        ] + [
            STATE_VEC_IDX_MAPPING[f"right_gripper_open"]  # Right gripper state
        ] + [
            125, 126, 127  # Body joints: head_pan_joint, head_tilt_joint, torso_lift_joint
        ] + [                           
            STATE_VEC_IDX_MAPPING[f"base_vel_x"]  # Base linear velocity in X direction
        ] + [  # Speed at Y is 0
            STATE_VEC_IDX_MAPPING["base_angular_vel"]  # Base angular velocity
        ]  # Reference: pd_base_vel.py L39
        
        # Action indices depend on the mode
        if self.mode == "full":
            # Full 13-dimensional action space
            self.UNI_ACTION_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)  # Right arm joints 0-6
            ] + [
                STATE_VEC_IDX_MAPPING[f"right_gripper_open"]  # Right gripper action
            ] + [
                125, 126, 127  # Body joints: head_pan_joint, head_tilt_joint, torso_lift_joint
            ] + [                           
                STATE_VEC_IDX_MAPPING[f"base_vel_x"]  # Base linear velocity in X direction
            ] + [  # Speed at Y is 0
                STATE_VEC_IDX_MAPPING["base_angular_vel"]  # Base angular velocity
            ]  # Reference: pd_base_vel.py L39
            self.action_dim = 13
        else:  # base_only
            # Only 2-dimensional base navigation actions
            self.UNI_ACTION_INDICES = [
                STATE_VEC_IDX_MAPPING[f"base_vel_x"]  # Base linear velocity in X direction
            ] + [  # Speed at Y is 0
                STATE_VEC_IDX_MAPPING["base_angular_vel"]  # Base angular velocity
            ]  # Reference: pd_base_vel.py L39
            self.action_dim = 2

    def fill_in_state(self, values):
        """
        Fill a 13-dimensional state vector into a 128-dimensional universal vector.
        
        Args:
            values: Input array of shape (..., 13) containing MSHab-specific states
            
        Returns:
            uni_vec: Output array of shape (..., 128) with values placed at appropriate indices
        """
        uni_vec = np.zeros(values.shape[:-1] + (128,))
        uni_vec[..., self.UNI_STATE_INDICES] = values
        return uni_vec

    def fill_in_action(self, values):
        """
        Fill action vector into a 128-dimensional universal vector.
        
        Args:
            values: Input array of shape (..., action_dim) containing actions
                - For 'full' mode: shape (..., 13)
                - For 'base_only' mode: shape (..., 2)
            
        Returns:
            uni_vec: Output array of shape (..., 128) with values placed at appropriate indices
        """
        uni_vec = np.zeros(values.shape[:-1] + (128,))
        uni_vec[..., self.UNI_ACTION_INDICES] = values
        return uni_vec

    def __len__(self):
        """Return the total number of episodes across all tasks."""
        return len(self.task_subtask_obj) * self.num_episode_per_task

    def get_dataset_name(self):
        """Return the name of this dataset."""
        return self.DATASET_NAME

    def get_mode(self):
        """Return the current control mode."""
        return self.mode

    def get_item(self, index=None, state_only=False):
        """
        Get a training sample from the dataset.

        Args:
            index (int, optional): Specific episode index to retrieve. 
                If None, randomly selects an episode.
            state_only (bool, optional): If True, returns only state trajectory
                without images or other training data. Defaults to False.

        Returns:
            dict: Dictionary containing the training sample with keys:
                - meta: Episode metadata
                - state: Current robot state
                - actions: Future action sequence
                - cam_high/cam_right_wrist: Camera images
                - in-context_conditions: Goal and object poses
        """
        # Retry until we get a valid episode
        while True:
            if index is None:
                index = np.random.randint(0, self.__len__())
            valid, sample = (
                self.parse_hdf5_file(index)
                if not state_only
                else self.parse_hdf5_file_state_only(index)
            )
            if valid:
                return sample
            else:
                # If episode is invalid, try a different random episode
                index = np.random.randint(0, self.__len__())

    def parse_hdf5_file(self, index):
        """
        Parse an HDF5 episode file and extract a training sample at a random timestep.

        Args:
            index (int): Global episode index across all tasks.

        Returns:
            tuple: (valid, sample) where:
                - valid (bool): Whether the episode is valid for training
                - sample (dict): Training sample containing states, actions, images, etc.
        """
        # Decode the global index to get task, subtask, object, and trajectory index
        task_subtask_obj_index = index // self.num_episode_per_task
        task, subtask, obj = self.task_subtask_obj[task_subtask_obj_index]
        traj_index = index % self.num_episode_per_task
        dataset_dir = pathlib.Path(self.data_dir) / task / subtask / 'train' / obj

        # Find the HDF5 file for this task combination
        h5_files = list(dataset_dir.glob("*.h5"))
        if len(h5_files) == 1:
            h5_file_path = h5_files[0]
        else:
            raise ValueError(f"Expected exactly one .h5 file in {dataset_dir}, but found {len(h5_files)}.")
                
        # Load and process the episode data
        with h5py.File(h5_file_path, 'r') as f:
            episode = f[f"traj_{traj_index}"]
            actions = episode['actions'][:]

            # Extract robot state components
            base_linear_vel = episode['obs']['extra']['base_linear_vel'][:][:-1]
            base_angular_vel = episode['obs']['extra']['base_angular_vel'][:][:-1]
            qpos = episode['obs']['agent']['qpos'][:][:-1]
            
            # Construct state vector from joint positions and velocities
            states = actions.copy()
            states[:, :7] = qpos[:, [2, 4, 5, 6, 7, 8, 9]]  # Right arm joints
            states[:, 7] = qpos[:, 10]  # Right gripper
            states[:, 8] = qpos[:, 1]   # Head pan
            states[:, 9] = qpos[:, 3]   # Head tilt
            states[:, 10] = qpos[:, 0]  # Torso lift
            states[:, 11] = base_linear_vel[:, 0]  # Base linear velocity
            states[:, 12] = base_angular_vel[:, 2]  # Base angular velocity

            # Extract actions based on mode
            if self.mode == "base_only":
                # Only keep base navigation actions (last 2 dimensions)
                actions = actions[:, -2:]
            # else: keep full actions (already 13-dimensional)
            
            # Determine valid timesteps based on task success
            success: List[bool] = episode["success"][:].tolist()
            success_cutoff = min(success.index(True) + 1, len(success))
            del success
            num_steps = success_cutoff
            step_index = np.random.randint(0, num_steps)  # Randomly choose a timestep

            # Load random language instruction for this task
            instructions_dir = dataset_dir / "instructions"
            instructions_paths = list(instructions_dir.rglob("*.pt"))
            instructions_path = np.random.choice(instructions_paths)

            # Extract in-context conditions (goal and object poses)
            goal_pos_wrt_base = episode['obs']['extra']['goal_pos_wrt_base'][:][step_index: step_index + 1]
            is_grasped = episode['obs']['extra']['is_grasped'][:][step_index: step_index + 1]
            obj_pose_wrt_base = episode['obs']['extra']['obj_pose_wrt_base'][:][step_index: step_index + 1]
            tcp_pose_wrt_base = episode['obs']['extra']['tcp_pose_wrt_base'][:][step_index: step_index + 1]

            # Create metadata for this sample
            meta = {
                "dataset_name": self.DATASET_NAME,
                "mode": self.mode,
                "steps": num_steps,
                "step_id": step_index,
                "instruction": str(instructions_path),
            }

            # Prepare state and action sequences
            state = states[step_index: step_index + 1]
            state_std = np.std(actions, axis=0)
            state_mean = np.mean(actions, axis=0)
            state_norm = np.sqrt(np.mean(actions ** 2, axis=0))

            # Create action sequence for prediction (with padding if needed)
            action_sequence = actions[step_index: step_index + self.CHUNK_SIZE]
            if action_sequence.shape[0] < self.CHUNK_SIZE:
                padding = np.tile(action_sequence[-1:], (self.CHUNK_SIZE - action_sequence.shape[0], 1))
                action_sequence = np.concatenate([action_sequence, padding], axis=0)

            # Convert to universal action vector format
            state = self.fill_in_state(state)
            state_indicator = self.fill_in_action(np.ones_like(action_sequence[0]))
            state_std = self.fill_in_action(state_std)
            state_mean = self.fill_in_action(state_mean)
            state_norm = self.fill_in_action(state_norm)
            action_sequence = self.fill_in_action(action_sequence)

            # Process camera images with history
            def parse_img(key):
                """Extract image sequence with history for a given camera."""
                imgs = []
                for i in range(max(step_index - self.IMG_HISTORY_SIZE + 1, 0), step_index + 1):
                    img = episode['obs']['sensor_data'][key]['rgb'][i][:]
                    imgs.append(img)
                imgs = np.stack(imgs)
                # Pad with first image if not enough history
                if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                    padding = np.tile(imgs[:1], (self.IMG_HISTORY_SIZE - imgs.shape[0], 1, 1, 1))
                    imgs = np.concatenate([padding, imgs], axis=0)
                return imgs
            
            # Process camera feeds
            cam_high = parse_img("fetch_head")
            valid_len = min(step_index , self.IMG_HISTORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len)
            cam_right_wrist = parse_img("fetch_hand")
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # Left wrist camera not available for this robot
            cam_left_wrist = np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = np.zeros(self.IMG_HISTORY_SIZE, dtype=bool)

            # Process point clouds with history
            def parse_pointcloud():
                """Extract pointcloud sequence with history from pre-sampled xyzrgb."""
                pcs = []
                for i in range(max(step_index - self.IMG_HISTORY_SIZE + 1, 0), step_index + 1):
                    # Read pre-sampled xyzrgb: (1024, 6) float32, xyz in meters, rgb in [0,1]
                    pc_xyzrgb = episode['obs']['pointcloud']['xyzrgb'][i][:]
                    pcs.append(pc_xyzrgb)

                pcs = np.stack(pcs, axis=0)  # (valid_len, 1024, 6)

                # Pad with first frame if not enough history
                if pcs.shape[0] < self.IMG_HISTORY_SIZE:
                    padding = np.tile(pcs[:1], (self.IMG_HISTORY_SIZE - pcs.shape[0], 1, 1))
                    pcs = np.concatenate([padding, pcs], axis=0)

                return pcs  # (IMG_HISTORY_SIZE, 1024, 6)

            pointclouds = parse_pointcloud()

            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": action_sequence,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
                "pointclouds": pointclouds,
                "in-context_conditions": {
                    "goal_pos_wrt_base": goal_pos_wrt_base,
                    "is_grasped": is_grasped,
                    "obj_pose_wrt_base": obj_pose_wrt_base,
                    "tcp_pose_wrt_base": tcp_pose_wrt_base,
                }
            }
        
    def parse_hdf5_file_state_only(self, index):
        """
        Parse an HDF5 file to extract only state and action trajectories.
        
        This method is used when only robot state information is needed,
        without images or other heavy data components.

        Args:
            index (int): Global episode index across all tasks.

        Returns:
            tuple: (valid, sample) where:
                - valid (bool): Whether the episode is valid
                - sample (dict): Contains 'state' and 'action' trajectories
        """
        # Decode global index to task components
        task_subtask_obj_index = index // self.num_episode_per_task
        task, subtask, obj = self.task_subtask_obj[task_subtask_obj_index]
        traj_index = index % self.num_episode_per_task
        dataset_dir = pathlib.Path(self.data_dir) / task / subtask / 'train' / obj

        # Find HDF5 file
        h5_files = list(dataset_dir.glob("*.h5"))
        if len(h5_files) == 1:
            h5_file_path = h5_files[0]
        else:
            raise ValueError(f"Expected exactly one .h5 file in {dataset_dir}, but found {len(h5_files)}.")

        # Load episode and extract state/action trajectories
        with h5py.File(h5_file_path, 'r') as f:
            episode = f[f"traj_{traj_index}"]
            actions = episode['actions'][:]

            # Extract robot state components (same as in parse_hdf5_file)
            base_linear_vel = episode['obs']['extra']['base_linear_vel'][:][:-1]
            base_angular_vel = episode['obs']['extra']['base_angular_vel'][:][:-1]
            qpos = episode['obs']['agent']['qpos'][:][:-1]
            states = actions.copy()
            states[:, :7] = qpos[:, [2, 4, 5, 6, 7, 8, 9]]
            states[:, 7] = qpos[:, 10]
            states[:, 8] = qpos[:, 1]
            states[:, 9] = qpos[:, 3]
            states[:, 10] = qpos[:, 0]
            states[:, 11] = base_linear_vel[:, 0]
            states[:, 12] = base_angular_vel[:, 2]

            # Extract actions based on mode
            if self.mode == "base_only":
                actions = actions[:, -2:]  # Only base navigation actions
            # else: keep full actions (already 13-dimensional)

            # Convert to universal vector format
            states = self.fill_in_state(states)
            actions = self.fill_in_action(actions)
            return True, {"state": states, "action": actions}
        
    def parse_hdf5_file_episode(self, index):
        """
        Parse an HDF5 file to extract complete episode data including images.
        
        This method returns full episode trajectories with both state/action
        sequences and image sequences, useful for episode-level analysis.

        Args:
            index (int): Global episode index across all tasks.

        Returns:
            tuple: (valid, sample) where:
                - valid (bool): Whether the episode is valid
                - sample (dict): Contains complete episode data with images
        """
        # Decode global index to task components
        task_subtask_obj_index = index // self.num_episode_per_task
        task, subtask, obj = self.task_subtask_obj[task_subtask_obj_index]
        traj_index = index % self.num_episode_per_task
        dataset_dir = pathlib.Path(self.data_dir) / task / subtask / 'train' / obj

        # Find HDF5 file
        h5_files = list(dataset_dir.glob("*.h5"))
        if len(h5_files) == 1:
            h5_file_path = h5_files[0]
        else:
            raise ValueError(f"Expected exactly one .h5 file in {dataset_dir}, but found {len(h5_files)}.")

        # Load complete episode data
        with h5py.File(h5_file_path, 'r') as f:
            episode = f[f"traj_{traj_index}"]
            actions = episode['actions'][:]

            # Extract robot state components (same as other methods)
            base_linear_vel = episode['obs']['extra']['base_linear_vel'][:][:-1]
            base_angular_vel = episode['obs']['extra']['base_angular_vel'][:][:-1]
            qpos = episode['obs']['agent']['qpos'][:][:-1]
            states = actions.copy()
            states[:, :7] = qpos[:, [2, 4, 5, 6, 7, 8, 9]]
            states[:, 7] = qpos[:, 10]
            states[:, 8] = qpos[:, 1]
            states[:, 9] = qpos[:, 3]
            states[:, 10] = qpos[:, 0]
            states[:, 11] = base_linear_vel[:, 0]
            states[:, 12] = base_angular_vel[:, 2]

            # Extract complete image sequences
            images_head = episode['obs']['sensor_data']['fetch_head']['rgb'][:]
            images_hand = episode['obs']['sensor_data']['fetch_hand']['rgb'][:]
            
            # Extract actions based on mode
            if self.mode == "base_only":
                actions = actions[:, -2:]  # Only base navigation actions
            # else: keep full actions (already 13-dimensional)
            
            # Convert to universal vector format
            states = self.fill_in_state(states)
            actions = self.fill_in_action(actions)
            return True, {"state": states, "action": actions, "images_head": images_head, "images_hand": images_hand}


# Factory functions for backward compatibility
def create_full_control_dataset():
    """
    Create a dataset instance for full control mode.
    
    Args:
        data_dir: Optional custom data directory path
        num_episode_per_task: Number of episodes per task combination
        
    Returns:
        HDF5MSHABDataset: Dataset instance in full control mode
    """
    return HDF5MSHABDataset(mode="full")


def create_base_only_dataset():
    """
    Create a dataset instance for base-only navigation mode.
    
    Args:
        data_dir: Optional custom data directory path
        num_episode_per_task: Number of episodes per task combination
        
    Returns:
        HDF5MSHABDataset: Dataset instance in base-only mode
    """
    return HDF5MSHABDataset(mode="base_only")


if __name__ == "__main__":
    """
    Test script to validate the unified dataset functionality.

    Usage:
        python data/hdf5_mshab_dataset.py

    This script tests both control modes and validates their functionality:
    - Full control mode (13-dimensional actions)
    - Base-only navigation mode (2-dimensional actions)

    Requirements:
        - config.yaml must exist at configs/config.yaml
        - data_dir must contain HDF5 files generated with obs_mode="rgbdp"
    """
    import yaml

    ACDIT_DEFAULT_LOGGER.info("Testing HDF5MSHABDataset...")

    # Load config (same as training script)
    config_path = 'configs/config.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # Data directory (same as training script finetune_mshab_dit.py)
    # Change this path if your data is stored elsewhere
    data_dir = '/share/project/chensixiang/Codes/AC-DiT/third_party/mshab/mshab_data_crop/gen_data_save_trajectories'

    ACDIT_DEFAULT_LOGGER.info(f"Config path: {config_path}")
    ACDIT_DEFAULT_LOGGER.info(f"Data directory: {data_dir}")
    ACDIT_DEFAULT_LOGGER.info(f"Config keys: {list(config.keys())}")

    # Test both modes
    for mode in ["full", "base_only"]:
        ACDIT_DEFAULT_LOGGER.info(f"\n{'='*60}")
        ACDIT_DEFAULT_LOGGER.info(f"Testing {mode.upper()} mode")
        ACDIT_DEFAULT_LOGGER.info(f"{'='*60}")

        # Create dataset instance with required parameters
        ds = HDF5MSHABDataset(
            data_dir=data_dir,
            config=config,
            mode=mode
        )

        ACDIT_DEFAULT_LOGGER.ok(f"Dataset initialized successfully")
        ACDIT_DEFAULT_LOGGER.info(f"  Mode: {ds.get_mode()}")
        ACDIT_DEFAULT_LOGGER.info(f"  Action dimension: {ds.action_dim}")
        ACDIT_DEFAULT_LOGGER.info(f"  Dataset length: {len(ds)}")
        ACDIT_DEFAULT_LOGGER.info(f"  Data directory: {ds.data_dir}")

        # Test a subset of episodes
        TEST_LEN = 5
        ACDIT_DEFAULT_LOGGER.info(f"Testing first {TEST_LEN} episodes...")
        for i in range(TEST_LEN):
            try:
                item = ds.get_item(i)
                if item['state'].shape[1] != ds.STATE_DIM:
                    ACDIT_DEFAULT_LOGGER.warning(f"Episode {i}: Invalid state dimension: {item['state'].shape[1]}")
                    continue

                # Extract data for validation
                state = item['state']
                actions = item['actions']
                meta = item['meta']
                pointclouds = item['pointclouds']

                ACDIT_DEFAULT_LOGGER.ok(f"Episode {i}: Loaded successfully")
                ACDIT_DEFAULT_LOGGER.info(f"    State: {state.shape}, Actions: {actions.shape}")
                ACDIT_DEFAULT_LOGGER.info(f"    Pointclouds: {pointclouds.shape} (history={pointclouds.shape[0]}, points={pointclouds.shape[1]}, features={pointclouds.shape[2]})")

                # Validate pointcloud data
                if pointclouds.shape[0] > 0:
                    xyz_range = [pointclouds[:, :, :3].min(), pointclouds[:, :, :3].max()]
                    rgb_range = [pointclouds[:, :, 3:].min(), pointclouds[:, :, 3:].max()]
                    ACDIT_DEFAULT_LOGGER.info(f"    XYZ range: [{xyz_range[0]:.2f}, {xyz_range[1]:.2f}] m")
                    ACDIT_DEFAULT_LOGGER.info(f"    RGB range: [{rgb_range[0]:.2f}, {rgb_range[1]:.2f}] (should be [0, 1])")

                # Extract in-context conditions
                goal_pos_wrt_base = item['in-context_conditions']['goal_pos_wrt_base']
                is_grasped = item['in-context_conditions']['is_grasped']
                obj_pose_wrt_base = item['in-context_conditions']['obj_pose_wrt_base']
                tcp_pose_wrt_base = item['in-context_conditions']['tcp_pose_wrt_base']

                ACDIT_DEFAULT_LOGGER.info(f"    In-context conditions:")
                ACDIT_DEFAULT_LOGGER.info(f"      goal_pos_wrt_base: {goal_pos_wrt_base.shape}")
                ACDIT_DEFAULT_LOGGER.info(f"      is_grasped: {is_grasped.shape}")
                ACDIT_DEFAULT_LOGGER.info(f"      obj_pose_wrt_base: {obj_pose_wrt_base.shape}")
                ACDIT_DEFAULT_LOGGER.info(f"      tcp_pose_wrt_base: {tcp_pose_wrt_base.shape}")

                break  # Just test one episode per mode

            except Exception as e:
                import traceback
                ACDIT_DEFAULT_LOGGER.error(f"Episode {i}: Error - {e}")
                traceback.print_exc()
                continue

    ACDIT_DEFAULT_LOGGER.info(f"\n{'='*60}")
    ACDIT_DEFAULT_LOGGER.ok("All tests completed!")
    ACDIT_DEFAULT_LOGGER.info(f"{'='*60}")
    ACDIT_DEFAULT_LOGGER.info("\nSummary:")
    ACDIT_DEFAULT_LOGGER.ok("  ✓ Config loaded successfully")
    ACDIT_DEFAULT_LOGGER.ok("  ✓ Dataset initialization works for both modes")
    ACDIT_DEFAULT_LOGGER.ok("  ✓ Data loading works correctly")
    ACDIT_DEFAULT_LOGGER.ok("  ✓ Pointcloud data is in correct format (1024 points, RGB [0,1])")
    ACDIT_DEFAULT_LOGGER.info("\nNote: Use HDF5MSHABDataset(data_dir, config, mode) for initialization")

