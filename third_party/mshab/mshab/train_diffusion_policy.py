import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import h5py
from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, RandomSampler

from mani_skill import ASSET_DIR
from mani_skill.utils import common

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from mshab.agents.dp import Agent
from mshab.agents.dp.utils import IterationBasedBatchSampler, worker_init_fn
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataclasses import default_field
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


@dataclass
class DPConfig:
    name: str = "diffusion_policy"

    # Diffusion Policy
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = default_field(
        [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are similar
    )
    encoded_image_feature_size: int = 1024

    # Dataset
    data_dir_fp: str = (
        ASSET_DIR
        / "scene_datasets/replica_cad_dataset/rearrange-dataset/tidy_hosue/pick"
    )
    """the path of demo dataset (dir or h5)"""
    trajs_per_obj: Union[Literal["all"], int] = "all"
    """number of trajectories to load from the demo dataset"""
    truncate_trajectories_at_success: bool = False
    """if true, truncate trajectories at first success"""
    max_image_cache_size: Union[Literal["all"], int] = 0
    """max num images to cache in cpu memory"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""

    # Experiment
    num_iterations: int = 1_000_000
    """total timesteps of the experiment"""
    eval_episodes: Optional[int] = None
    """the number of episodes to evaluate the agent on"""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: int = 5000
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    save_backup_ckpts: bool = False


@dataclass
class TrainConfig:
    seed: int
    eval_env: EnvConfig
    algo: DPConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        # assert self.eval_env.frame_stack is None and isinstance(
        #     self.eval_env.stack, int
        # )
        # assert self.eval_env.stack == self.algo.obs_horizon

        if self.algo.eval_episodes is None:
            self.algo.eval_episodes = self.eval_env.num_envs
        self.algo.eval_episodes = max(self.algo.eval_episodes, self.eval_env.num_envs)
        assert self.algo.eval_episodes % self.eval_env.num_envs == 0

        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "Can't resume to a cleared out logdir!"

        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


def recursive_h5py_to_numpy(h5py_obs, slice=None):
    if isinstance(h5py_obs, h5py.Group) or isinstance(h5py_obs, dict):
        return dict(
            (k, recursive_h5py_to_numpy(h5py_obs[k], slice)) for k in h5py_obs.keys()
        )
    if isinstance(h5py_obs, list):
        return [recursive_h5py_to_numpy(x, slice) for x in h5py_obs]
    if isinstance(h5py_obs, tuple):
        return tuple(recursive_h5py_to_numpy(x, slice) for x in h5py_obs)
    if slice is not None:
        return h5py_obs[slice]
    return h5py_obs[:]


class DPDataset(ClosableDataset):  # Load everything into memory
    def __init__(
        self,
        data_path,
        obs_horizon,
        pred_horizon,
        control_mode,
        trajs_per_obj="all",
        max_image_cache_size=0,
        truncate_trajectories_at_success=False,
    ):
        data_path = Path(data_path)
        if data_path.is_dir():
            h5_fps = [
                data_path / fp for fp in os.listdir(data_path) if fp.endswith(".h5")
            ]
        else:
            h5_fps = [data_path]

        trajectories = dict(actions=[], observations=[])
        num_cached = 0
        self.h5_files = []
        for fp_num, fp in enumerate(h5_fps):
            f = h5py.File(fp, "r")
            num_uncached_this_file = 0

            if trajs_per_obj == "all":
                keys = list(f.keys())
            else:
                keys = random.sample(list(f.keys()), k=trajs_per_obj)

            for k in tqdm(keys, desc=f"hf file {fp_num}"):
                obs, act = f[k]["obs"], f[k]["actions"][:]

                if truncate_trajectories_at_success:
                    success: List[bool] = f[k]["success"][:].tolist()
                    success_cutoff = min(success.index(True) + 1, len(success))
                    del success
                else:
                    success_cutoff = len(act)

                # NOTE (arth): we always cache state obs and actions because they take up very little memory.
                #       mostly constraints are on images, since those take up much more memory
                state_obs_list = [
                    *recursive_h5py_to_numpy(
                        obs["agent"], slice=slice(success_cutoff + 1)
                    ).values(),
                    *recursive_h5py_to_numpy(
                        obs["extra"], slice=slice(success_cutoff + 1)
                    ).values(),
                ]
                state_obs_list = [
                    x[:, None] if len(x.shape) == 1 else x for x in state_obs_list
                ]
                state_obs = torch.from_numpy(np.concatenate(state_obs_list, axis=1))
                # remove the base angular vel, base linear vel and base pose since it is not needed in training 
                state_obs = state_obs[:, :-9]
                # don't cut off actions in case we are able to use in place of padding
                act = torch.from_numpy(act)

                pixel_obs = dict(
                    fetch_head_depth=obs["sensor_data"]["fetch_head"]["depth"],
                    fetch_hand_depth=obs["sensor_data"]["fetch_hand"]["depth"],
                )
                if (
                    max_image_cache_size == "all"
                    or len(act) <= max_image_cache_size - num_cached
                ):
                    pixel_obs = to_tensor(
                        recursive_h5py_to_numpy(
                            pixel_obs, slice=slice(success_cutoff + 1)
                        )
                    )
                    num_cached += len(act)
                else:
                    num_uncached_this_file += len(act)

                trajectories["actions"].append(act)
                trajectories["observations"].append(dict(state=state_obs, **pixel_obs))

            if num_uncached_this_file == 0:
                f.close()
            else:
                self.h5_files.append(f)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in control_mode
            or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,)
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            raise NotImplementedError(f"Control Mode {control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            obs_horizon,
            pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            # NOTE (arth): since we cut off data at first success, we might have extra actions available
            #   after the end of slice which we can use instead of hand-made padded zero actions
            L = trajectories["observations"][traj_idx]["state"].shape[0] - 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if len(obs_seq[k].shape) == 4:
                obs_seq[k] = to_tensor(obs_seq[k]).permute(0, 3, 2, 1)  # FS, D, H, W
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)

    def close(self):
        for h5_file in self.h5_files:
            h5_file.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))

    print("cfg:", cfg, flush=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    assert cfg.algo.obs_horizon + cfg.algo.act_horizon - 1 <= cfg.algo.pred_horizon
    assert (
        cfg.algo.obs_horizon >= 1
        and cfg.algo.act_horizon >= 1
        and cfg.algo.pred_horizon >= 1
    )

    # NOTE (arth): maybe require cuda since we only allow gpu sim anyways
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------------------------------
    # ENVS
    # -------------------------------------------------------------------------------------------------

    print("making eval env", flush=True)
    eval_envs = make_env(
        cfg.eval_env,
        video_path=cfg.logger.eval_video_path,
    )
    assert isinstance(
        eval_envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)

    print("made", flush=True)

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------

    print("making agent and logger...", flush=True)

    agent = Agent(
        single_observation_space=eval_envs.single_observation_space,
        single_action_space=eval_envs.single_action_space,
        obs_horizon=cfg.algo.obs_horizon,
        act_horizon=cfg.algo.act_horizon,
        pred_horizon=cfg.algo.pred_horizon,
        diffusion_step_embed_dim=cfg.algo.diffusion_step_embed_dim,
        unet_dims=cfg.algo.unet_dims,
        n_groups=cfg.algo.n_groups,
        device=device,
    ).to(device)

    optimizer = optim.AdamW(
        params=agent.parameters(),
        lr=cfg.algo.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=cfg.algo.num_iterations,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(
        single_observation_space=eval_envs.single_observation_space,
        single_action_space=eval_envs.single_action_space,
        obs_horizon=cfg.algo.obs_horizon,
        act_horizon=cfg.algo.act_horizon,
        pred_horizon=cfg.algo.pred_horizon,
        diffusion_step_embed_dim=cfg.algo.diffusion_step_embed_dim,
        unet_dims=cfg.algo.unet_dims,
        n_groups=cfg.algo.n_groups,
        device=device,
    ).to(device)

    # -------------------------------------------------------------------------------------------------
    # LOGGER
    # -------------------------------------------------------------------------------------------------

    def save(save_path):
        ema.copy_to(ema_agent.parameters())
        torch.save(
            {
                "agent": agent.state_dict(),
                "ema_agent": ema_agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            save_path,
        )

    def load(load_path):
        checkpoint = torch.load(load_path)
        agent.load_state_dict(checkpoint["agent"])
        ema_agent.load_state_dict(checkpoint["ema_agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,
    )

    if cfg.model_ckpt is not None:
        load(cfg.model_ckpt)

    print("agent and logger made!", flush=True)

    # -------------------------------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------------------------------

    print("loading dataset...", flush=True)

    dataset = DPDataset(
        cfg.algo.data_dir_fp,
        obs_horizon=cfg.algo.obs_horizon,
        pred_horizon=cfg.algo.pred_horizon,
        control_mode=eval_envs.unwrapped.control_mode,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_image_cache_size,
        truncate_trajectories_at_success=cfg.algo.truncate_trajectories_at_success,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(
        sampler, batch_size=cfg.algo.batch_size, drop_last=True
    )
    batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.algo.num_iterations)
    train_dataloader = ClosableDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.algo.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=cfg.seed),
        pin_memory=True,
        persistent_workers=(cfg.algo.num_dataload_workers > 0),
    )

    print("dataset loaded!", flush=True)

    # -------------------------------------------------------------------------------------------------
    # STARTING
    # -------------------------------------------------------------------------------------------------

    iteration = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0

    def check_freq(freq):
        return iteration % freq == 0

    def store_env_stats(key):
        log_env = eval_envs
        success_once=common.to_tensor(log_env.success_once_queue, device=device).float().mean()
        cprint(f"Success Once: {success_once.item()}", "green")        
        logger.store(
            key,
            return_per_step=common.to_tensor(log_env.return_queue, device=device)
            .float()
            .mean()
            / log_env.max_episode_steps,
            success_once=common.to_tensor(log_env.success_once_queue, device=device)
            .float()
            .mean(),
            success_at_end=common.to_tensor(log_env.success_at_end_queue, device=device)
            .float()
            .mean(),
            len=common.to_tensor(log_env.length_queue, device=device).float().mean(),
        )
        extra_stat_logs = dict()
        for k, v in log_env.extra_stats.items():
            extra_stat_values = torch.stack(v)
            extra_stat_logs[f"{k}_once"] = torch.mean(
                torch.any(extra_stat_values, dim=1).float()
            )
            extra_stat_logs[f"{k}_at_end"] = torch.mean(
                extra_stat_values[..., -1].float()
            )
        logger.store(f"extra/{key}", **extra_stat_logs)
        log_env.reset_queues()

    agent.train()

    timer = NonOverlappingTimeProfiler()

    for iteration, data_batch in tqdm(
        enumerate(train_dataloader),
        initial=logger_start_log_step,
        total=cfg.algo.num_iterations,
    ):
        data_batch = to_tensor(data_batch, device=device, dtype=torch.float)
        if iteration + logger_start_log_step > cfg.algo.num_iterations:
            break

        # # copy data from cpu to gpu
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        # forward and compute loss
        obs_batch_dict = data_batch["observations"]
        obs_batch_dict = {
            k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()
        }
        act_batch = data_batch["actions"].cuda(non_blocking=True)

        # forward and compute loss
        total_loss = agent.compute_loss(
            obs_seq=obs_batch_dict,  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=act_batch,  # (B, L, act_dim)
        )

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior

        ema.step(agent.parameters())

        logger.store("losses", loss=total_loss.item())
        logger.store("charts", learning_rate=optimizer.param_groups[0]["lr"])
        timer.end(key="train")

        if check_freq(cfg.algo.log_freq):
            if iteration > 0:
                logger.store("time", **timer.get_time_logs(iteration))
            logger.log(logger_start_log_step + iteration)
            timer.end("log")

        # Evaluation
        if cfg.algo.eval_freq:
            if check_freq(cfg.algo.eval_freq):
                with torch.no_grad():
                    ema.copy_to(ema_agent.parameters())
                    agent.eval()
                    obs, info = eval_envs.reset()
                    while len(eval_envs.return_queue) < cfg.algo.eval_episodes:
                        obs = common.to_tensor(obs, device)
                        action_seq = agent.get_action(obs)
                        for i in range(action_seq.shape[1]):
                            obs, rew, terminated, truncated, info = eval_envs.step(
                                action_seq[:, i]
                            )
                            if truncated.any():
                                break

                        if truncated.any():
                            assert (
                                truncated.all() == truncated.any()
                            ), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                    agent.train()
                    if len(eval_envs.return_queue) > 0:
                        store_env_stats("eval")
                    logger.log(logger_start_log_step + iteration)
                    timer.end(key="eval")

        # Checkpoint
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{iteration}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end(key="checkpoint")

    train_dataloader.close()
    eval_envs.close()
    logger.close()
