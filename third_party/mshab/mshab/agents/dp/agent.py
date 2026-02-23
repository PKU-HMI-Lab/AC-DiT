from typing import List

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mshab.agents.dp.conditional_unet1d import ConditionalUnet1D
from mshab.agents.dp.plain_conv import PlainConv


class Agent(nn.Module):
    def __init__(
        self,
        single_observation_space: gym.spaces.Dict,
        single_action_space: gym.spaces.Box,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        diffusion_step_embed_dim: int,
        unet_dims: List[int],
        n_groups: int,
        device: torch.device,
    ):
        super().__init__()

        self.device = device

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon

        self.image_keys = [k for k in single_observation_space.keys() if k != "state"]
        self.depth_keys = [k for k in self.image_keys if "depth" in k]

        assert (
            len(single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(single_action_space.shape) == 1  # (act_dim, )
        assert np.all(single_action_space.high == 1) and np.all(
            single_action_space.low == -1
        )
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = single_action_space.shape[0]
        obs_state_dim = single_observation_space["state"].shape[1]

        # TODO (arth): support separate image encoders for each image key
        #   for now we will manually concat images together in Agent
        in_c = 0
        for imk in self.image_keys:
            fs, d, h, w = single_observation_space[imk].shape
            in_c += d
        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            img_shape=(in_c, h, w), out_dim=visual_feature_dim, pool_feature_map=True
        )

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        # NOTE (arth): we don't need to permute images since our wrappers already give shape (B, FS, D, H, W)
        # obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)

        # NOTE (arth): i guess dividing by 1024 helps with depth in diffusion policy?
        #   dividing is fine for nomalization since it preseves objective distance information (unlike min-max normalization)
        for dk in self.depth_keys:
            # obs_seq[dk] = obs_seq[dk].float() / 1024
            obs_seq[dk] = 1 - torch.tanh(obs_seq[dk].float() / 1000)
        # if args.depth:
        #     obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3) / 1024

        img_seq = torch.cat(
            [obs_seq[k] for k in self.image_keys], dim=2
        )  # (B, obs_horizon, C, H, W), for Fetch C=2 (since we have 2 depth images)
        B = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            B, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)
