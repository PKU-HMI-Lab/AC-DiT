import os
import pathlib
import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from lift3d.models.lift3d.model_loader import Lift3dSigLip

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import DiT


class DiTRunner(
    nn.Module,
    CompatiblePyTorchModelHubMixin,
    repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b",
):
    def __init__(
        self,
        *,
        action_dim,
        pred_horizon,
        config,
        lang_token_dim,
        img_token_dim,
        pc_token_dim,
        state_token_dim,
        max_lang_cond_len,
        img_cond_len,
        pc_cond_len,
        in_context_cond_dim,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super(DiTRunner, self).__init__()
        # Create diffusion model
        hidden_size = config["rdt"]["hidden_size"]
        self.model = DiT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config["rdt"]["depth"],
            num_heads=config["rdt"]["num_heads"],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            pc_cond_len=pc_cond_len,
            dtype=dtype,
        )
        self.lift3d = Lift3dSigLip()
        self.lift3d.to(dtype=dtype)

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config["lang_adaptor"], in_features=lang_token_dim, out_features=hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'],
            in_features=img_token_dim,
            out_features=hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config["state_adaptor"],
            in_features=state_token_dim * 2,  # state + state mask (indicator)
            out_features=hidden_size,
        )
        self.lift3d_adaptor = self.build_condition_adapter(
            config["lift3d_adaptor"], in_features=pc_token_dim, out_features=hidden_size
        )
        
        # Only create in_context_conditions_adaptor if in_context_cond_dim > 0
        self.in_context_cond_dim = in_context_cond_dim
        if in_context_cond_dim > 0:
            self.in_context_conditions_adaptor = self.build_condition_adapter(
                config["in_context_conditions_adaptor"],  # fallback to goal_pos config
                in_features=in_context_cond_dim, 
                out_features=hidden_size,
            )
        else:
            self.in_context_conditions_adaptor = None

        # Create the noise scheduler
        noise_scheduler_config = config["noise_scheduler"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_schedule=noise_scheduler_config["beta_schedule"],
            prediction_type=noise_scheduler_config["prediction_type"],
            clip_sample=noise_scheduler_config["clip_sample"],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_schedule=noise_scheduler_config["beta_schedule"],
            prediction_type=noise_scheduler_config["prediction_type"],
        )

        self.num_train_timesteps = noise_scheduler_config["num_train_timesteps"]
        self.num_inference_timesteps = noise_scheduler_config["num_inference_timesteps"]
        self.prediction_type = noise_scheduler_config["prediction_type"]

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        print(
            "Diffusion params: %e"
            % sum(
                [p.numel() for p in self.model.parameters()]
                + [p.numel() for p in self.lang_adaptor.parameters()]
                + [p.numel() for p in self.img_adaptor.parameters()]
                + [p.numel() for p in self.state_adaptor.parameters()]
                + [p.numel() for p in self.lift3d_adaptor.parameters()]
                + ([p.numel() for p in self.in_context_conditions_adaptor.parameters()] 
                   if self.in_context_conditions_adaptor is not None else [])
            )
        ) 

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        config: Dict,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        # Extract model config from full config
        model_config = config["model"]

        if "pc_token_dim" not in model_kwargs:
            model_kwargs["pc_token_dim"] = model_config["pc_token_dim"]
        if "pc_cond_len" not in model_kwargs:
            model_kwargs["pc_cond_len"] = (
                config["common"]["img_history_size"]
                * config["dataset"]["pointcloud_feature_length"]
            )
        if "config" not in model_kwargs:
            model_kwargs["config"] = model_config
        if "lift3d_adaptor" not in model_kwargs["config"]:
            model_kwargs["config"]["lift3d_adaptor"] = model_config["lift3d_adaptor"]

        # Add unified in-context conditions adapter configuration (backward compatibility)
        if "in_context_conditions_adaptor" not in model_kwargs["config"]:
            # Use goal_pos_adaptor config as fallback for the unified adapter
            model_kwargs["config"]["in_context_conditions_adaptor"] = model_config["in_context_conditions_adaptor"]

        model_kwargs['pred_horizon'] = config["common"]["action_chunk_size"]

        model_kwargs["lang_token_dim"] = model_config["lang_token_dim"]

        model = cls(**model_kwargs)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            try:
                model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except FileNotFoundError:
                model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
                return cls._load_as_pickle(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )

                state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
        
                if hasattr(model, "lang_adaptor") and hasattr(model.lang_adaptor[0], "weight"):
                    if model.lang_adaptor[0].weight.shape[1] == 1152:
                        keys_to_remove = []
                        for key in state_dict.keys():
                            if key.startswith("lang_adaptor"):
                                keys_to_remove.append(key)
                        for key_to_remove in keys_to_remove:
                            print(f"Removing key '{key_to_remove}' from state_dict due to shape mismatch.")
                            del state_dict[key_to_remove]

                model.load_state_dict(state_dict, strict=strict)  # type: ignore
                model.eval()  # type: ignore
                return model

    def encode_pointcloud(self, pointclouds, weight_dtype):
        batch_size, T, N, D = pointclouds.shape
        pointclouds = pointclouds.reshape(batch_size * T, N, D)
        pc_tokens = self.lift3d(pointclouds)
        pc_tokens = pc_tokens.reshape(batch_size, T, pc_tokens.shape[-2], pc_tokens.shape[-1])
        pc_tokens = pc_tokens.reshape(batch_size, T * pc_tokens.shape[-2], pc_tokens.shape[-1])
        pc_tokens = pc_tokens.to(dtype=weight_dtype)
        return pc_tokens

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == "linear":
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f"Unknown projector type: {projector_type}")

        return projector

    def adapt_conditions(
        self, 
        lang_tokens, 
        img_tokens, 
        pc_tokens, 
        state_tokens,
        in_context_conditions_tokens,
    ):
        """
        Adapt various conditional inputs to hidden dimension.
        
        Args:
            lang_tokens: (batch_size, lang_len, lang_token_dim) - Language tokens
            img_tokens: (batch_size, img_len, img_token_dim) - Image tokens
            pc_tokens: Point cloud tokens
            state_tokens: (batch_size, state_len, state_token_dim) - State tokens
            in_context_conditions_tokens: (batch_size, 1, 18) - Unified in-context conditions
                containing concatenated goal_pos (3), is_grasped (1), obj_pose (7), tcp_pose (7)

        Returns:
            tuple: Adapted tokens with shape (..., hidden_size) for all input tokens
        """
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_pc = self.lift3d_adaptor(pc_tokens)
        adapted_state = self.state_adaptor(state_tokens)
        
        if self.in_context_conditions_adaptor is not None:
            adapted_in_context_conditions = self.in_context_conditions_adaptor(in_context_conditions_tokens)
        else:
            # Create dummy tensor with correct shape when no in-context conditions
            batch_size = lang_tokens.shape[0]
            device = lang_tokens.device
            dtype = lang_tokens.dtype
            adapted_in_context_conditions = torch.zeros(
                (batch_size, 1, self.model.hidden_size), 
                device=device, 
                dtype=dtype
            )

        return adapted_lang, adapted_img, adapted_pc, adapted_state, adapted_in_context_conditions

    def conditional_sample(
        self,
        lang_cond,
        lang_attn_mask,
        img_cond,
        pc_cond,
        state_traj,
        action_mask,
        ctrl_freqs,
        in_context_conditions_cond,
    ):
        """
        Conditional sampling for action prediction.
        
        Args:
            lang_cond: (batch_size, lang_len, hidden_size) - Language conditional data
            lang_attn_mask: (batch_size, lang_len) - Language attention mask (True-False bool tensor)
            img_cond: (batch_size, img_len, hidden_size) - Image conditional data
            pc_cond: Point cloud conditional data
            state_traj: (batch_size, 1, hidden_size) - State trajectory
            action_mask: (batch_size, 1, action_dim) - Action mask (0-1 float tensor)
            ctrl_freqs: (batch_size,) - Control frequency for each sample
            in_context_conditions_cond: (batch_size, 1, hidden_size) - Unified in-context conditions

        Returns:
            torch.Tensor: (batch_size, horizon, action_dim) - Predicted actions
        """
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # Predict the model output
            model_output = self.model(
                x=state_action_traj,
                freq=ctrl_freqs,
                t=t.unsqueeze(-1).to(device),
                lang_c=lang_cond,
                img_c=img_cond,
                pc_c=pc_cond,
                lang_mask=lang_attn_mask,
                in_context_conditions_c=in_context_conditions_cond,
            )

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action
            ).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action

    # ========= Train  ============
    def compute_loss(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        pc_tokens, 
        state_tokens,
        action_gt,
        action_mask,
        ctrl_freqs,
        in_context_conditions,
    ) -> torch.Tensor:
        """
        Compute training loss for the model.
        
        Args:
            lang_tokens: (batch_size, lang_len, lang_token_dim) - Language tokens
            lang_attn_mask: (batch_size, lang_len) - Language attention mask (True-False bool tensor)
            img_tokens: (batch_size, img_len, img_token_dim) - Image tokens
            pc_tokens: Point cloud tokens
            state_tokens: (batch_size, 1, state_token_dim) - State tokens
            action_gt: (batch_size, horizon, state_token_dim) - Ground-truth actions for supervision
            action_mask: (batch_size, 1, state_token_dim) - Action mask (0-1 float tensor)
            ctrl_freqs: (batch_size,) - Control frequency for each sample
            in_context_conditions: (batch_size, 1, 18) - Unified in-context conditions tensor
                containing concatenated goal_pos (3), is_grasped (1), obj_pose (7), tcp_pose (7)

        Returns:
            torch.Tensor: Loss value (scalar tensor)
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # Sample noise that we'll add to the actions
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)

        # Align the dimension with the hidden size
        lang_cond, img_cond, pc_cond, state_action_traj, \
        in_context_conditions_cond = self.adapt_conditions(
            lang_tokens=lang_tokens,
            img_tokens=img_tokens,
            pc_tokens=pc_tokens,
            state_tokens=state_action_traj,
            in_context_conditions_tokens=in_context_conditions,
        ) # * add pc_cond

        # Predict the denoised result
        pred = self.model(
            x=state_action_traj,
            freq=ctrl_freqs,
            t=timesteps,
            lang_c=lang_cond,
            img_c=img_cond,
            pc_c=pc_cond,
            lang_mask=lang_attn_mask,
            in_context_conditions_c=in_context_conditions_cond,
        )

        pred_type = self.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)

        loss_dict = {
            "loss": loss,
        }

        return loss_dict

    # ========= Inference  ============
    def predict_action(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        pc_tokens,
        state_tokens,
        action_mask,
        ctrl_freqs,
        in_context_conditions,
    ):
        """
        Predict action sequence given current state and conditions.
        
        Args:
            lang_tokens: (batch_size, lang_len, lang_token_dim) - Language tokens
            lang_attn_mask: (batch_size, lang_len) - Language attention mask (True-False bool tensor)
            img_tokens: (batch_size, img_len, img_token_dim) - Image tokens
            pc_tokens: Point cloud tokens
            state_tokens: (batch_size, 1, state_token_dim) - State tokens
            action_mask: (batch_size, 1, action_dim) - Action mask (0-1 float tensor)
            ctrl_freqs: (batch_size,) - Control frequency for each sample
            in_context_conditions: (batch_size, 1, 18) - Unified in-context conditions tensor
                containing concatenated goal_pos (3), is_grasped (1), obj_pose (7), tcp_pose (7)

        Returns:
            torch.Tensor: (batch_size, horizon, action_dim) - Predicted action sequence
        """
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, pc_cond, state_traj, \
        in_context_conditions_cond = self.adapt_conditions(
            lang_tokens=lang_tokens,
            img_tokens=img_tokens,
            pc_tokens=pc_tokens,
            state_tokens=state_tokens,
            in_context_conditions_tokens=in_context_conditions,
        ) 

        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            pc_cond,
            state_traj,
            action_mask,
            ctrl_freqs,
            in_context_conditions_cond,
        )

        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
