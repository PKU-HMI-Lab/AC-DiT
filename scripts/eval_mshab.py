from typing import Callable, List, Type
import sys
import random
import os
sys.path.append('/')
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
import tqdm
from model_wrappers.mshab_model import create_model, MSHabModel
import torch
from collections import deque
from PIL import Image
import cv2
import imageio
import pathlib
from helpers.common import ACDIT_DEFAULT_LOGGER
from mshab.utils.debug import recursive_print_dict
import mshab.envs
from mshab.envs.planner import plan_data_from_file


ASSET_DIR = pathlib.Path(os.environ['MS_ASSET_DIR'])
REARRANGE_DIR = ASSET_DIR / "data/scene_datasets/replica_cad_dataset/rearrange"
METHODS = ["DiT", "AC-DiT"]


'''
Example usage:

python -m scripts.eval_mshab \
    --method-name AC-DiT \
    --task set_table \
    --subtask pick \
    --object 013_apple \
    --num-traj 10 \
    --pretrained-path {path_to_the_pretrained_model} \
    --mobility-head-ckpt-path {path_to_the_mobility_head_checkpoint} \
    --dataset-dir {path_to_the_dataset} \
    --result-dir {path_to_the_result_directory} \
    --control-mode pd_joint_delta_pos \
    --overwrite
'''

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", type=str, default="AC-DiT", choices=METHODS, help="Method name to use. Currently only RDT is supported.")
    parser.add_argument("-e", "--env", type=str, default="pick_bowl", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("--task", type=str, default="set_table", choices=["set_table", "tidy_house", "prepare_groceries"], help="Task name to use.")
    parser.add_argument("--subtask", type=str, default="pick", choices=["pick", "place", "open", "close"], help="Subtask name to use.")
    parser.add_argument("--object", type=str, default="013_apple", choices=["013_apple", "024_bowl", "fridge", "kitchen_counter"], help="Object name to use.")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgbdp", help="Observation mode to use. rgbdp returns both raw images and pre-sampled pointcloud.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to test.")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos", choices=["pd_joint_delta_pos", "pd_joint_pos"], help="Control mode to use. Usually this is kept as 'pd_joint_delta_pos' as this is the default control mode for the fetch robot.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str, default="dense")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument('--params-170m', action='store_true', help='Use 170m model')
    parser.add_argument("--dataset-dir", type=str, default="./")
    parser.add_argument("--result-dir", type=str, default="./eval_rdt_mshab")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--combine-flag", action="store_true")
    parser.add_argument(
        "--mobility-head-ckpt-path",
        type=str,
        default=None,
        help="Path to the mobility head checkpoint file (only used when method-name is AC-DiT). If None, uses the default checkpoint path.",
    )
    return parser.parse_args()


def get_proprio(obs: dict) -> torch.Tensor:
    qpos = obs['agent']['qpos']
    proprio = torch.zeros((1, 13), device=qpos.device, dtype=qpos.dtype)
    proprio[0, 0] = qpos[0, 2]
    proprio[0, 1:7] = qpos[0, 4:10]
    proprio[0, 7] = qpos[0, 10]
    proprio[0, 8] = qpos[0, 1]
    proprio[0, 9] = qpos[0, 3]
    proprio[0, 10] = qpos[0, 0]
    proprio[0, 11] = obs['extra']['base_linear_vel'][0, 0]
    proprio[0, 12] = obs['extra']['base_angular_vel'][0, 2]
    return proprio


def get_pointcloud(obs):
    """Extract pre-sampled pointcloud from obs (obs_mode='rgbdp')."""
    # obs['pointcloud']['xyzrgb']: (num_envs, 1024, 6) float32
    # xyz in meters, rgb in [0, 1]
    xyzrgb = obs['pointcloud']['xyzrgb'].cpu().numpy().squeeze(0)  # (1024, 6)
    return xyzrgb


def main(args):
    # set random seeds
    seed = args.random_seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    result_dir = pathlib.Path(args.result_dir) / f"{args.task}_{args.subtask}_{args.object}"
    subtask = args.subtask
    task_name = args.task
    plan_obj_name = args.object

    result_dir.mkdir(parents=True, exist_ok=True)
    success_rate_path = result_dir / 'success_rate.txt'
    if pathlib.Path(success_rate_path).exists():
        if args.overwrite:
            ACDIT_DEFAULT_LOGGER.warning(f"{success_rate_path} exists, overwriting...")
        else:
            ACDIT_DEFAULT_LOGGER.warning(f"{success_rate_path} exists, skipping...")
            exit(0)

    # get env
    env_id = f'{subtask.capitalize()}SubtaskTrain-v0'
    split = "val"
    
    plan_data = plan_data_from_file(
    REARRANGE_DIR / "task_plans" / task_name / subtask / split / f"{plan_obj_name}.json")
    spawn_data_fp = REARRANGE_DIR / "spawn_data" / task_name / subtask / split / "spawn_data.pt"
    env = gym.make(
        env_id,
        num_envs=1,
        robot_uids="fetch",
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        reward_mode=args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        max_episode_steps=200,
        task_plans=plan_data.plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False
    )

    # load model
    config_path = 'configs/config.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
        if args.params_170m:
            config['model']['rdt']['hidden_size'] = 1024
            config['model']['rdt']['depth'] = 14
            config['model']['rdt']['num_heads'] = 32
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    pretrained_path = args.pretrained_path
    policy = create_model(
        args=config,
        method_name=args.method_name,
        device=args.device,
        dtype=torch.bfloat16,
        pretrained=pretrained_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        combine_flag=args.combine_flag,
        mobility_head_ckpt_path=args.mobility_head_ckpt_path,
    )

    MAX_EPISODE_STEPS = 200 # 200 steps for MSHAB task 
    total_episodes = args.num_traj  
    success_count = 0  

    base_seed = 2024
    progress_bar = tqdm.trange(total_episodes)
    for episode in progress_bar:
        # load language embedding
        policy.random_set_language(dataset_dir=args.dataset_dir)

        obs_window = deque(maxlen=2)
        obs_pc_window = deque(maxlen=2)
        if episode == 0:
            obs, _ = env.reset(seed=base_seed)
        else:
            obs, _ = env.reset(seed=None)
        policy.reset()

        head_img = obs['sensor_data']['fetch_head']['rgb'].cpu().numpy().squeeze(0)
        hand_img = obs['sensor_data']['fetch_hand']['rgb'].cpu().numpy().squeeze(0)
        pointcloud = get_pointcloud(obs)
        obs_window.append([None, None, None])
        obs_window.append([head_img, hand_img, None])
        obs_pc_window.append(None)
        obs_pc_window.append(pointcloud)
        
        proprio = get_proprio(obs)
        
        global_steps = 0
        video_frames_head, video_frames_hand = [], []

        done = False

        while global_steps < MAX_EPISODE_STEPS and not done:
            image_arrs = []
            for window_imgs in obs_window:  
                image_arrs.append(window_imgs[0]) 
                image_arrs.append(window_imgs[1])
                image_arrs.append(window_imgs[2])
            images = [Image.fromarray(arr) if arr is not None else None
                    for arr in image_arrs]

            pointclouds = [obs_pc_window[0], obs_pc_window[1]]
            
            actions = policy.step(proprio, images, policy.lang_embeddings, pointclouds, obs['extra']).squeeze(0).cpu().numpy()

            for idx in range(actions.shape[0]):
                progress_bar.set_description(f"Step : {global_steps} / {MAX_EPISODE_STEPS}")
                action = actions[idx]
                obs, reward, terminated, truncated, info = env.step(action)
                head_img = obs['sensor_data']['fetch_head']['rgb'].cpu().numpy().squeeze(0)
                hand_img = obs['sensor_data']['fetch_hand']['rgb'].cpu().numpy().squeeze(0)
                pointcloud = get_pointcloud(obs)
                obs_window.append([head_img, hand_img, None])
                obs_pc_window.append(pointcloud)
                proprio = get_proprio(obs)
                video_frames_head.append(head_img)
                video_frames_hand.append(hand_img)
                global_steps += 1
                if terminated or truncated:
                    assert "success" in info, sorted(info.keys())
                    if info['success']:
                        success_count += 1
                        done = True
                        break 
        print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")

        video_path = result_dir / f"trial_{episode + 1}_{'success' if info['success'] else 'failure'}.mp4"
        with imageio.get_writer(video_path, mode='I', fps=10, codec='libx264') as writer:
            for head_frame, hand_frame in zip(video_frames_head, video_frames_hand):
                combined_frame = np.concatenate((head_frame, hand_frame), axis=1)
                writer.append_data(combined_frame)
        print(f"save video to {video_path}")

    success_rate = success_count / total_episodes * 100
    print(f"Success rate: {success_rate}%")
    with open(success_rate_path, 'w') as f:
        f.write(f"Success rate: {success_rate}%\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
