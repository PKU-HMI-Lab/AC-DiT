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

from helpers.common import ACDIT_DEFAULT_LOGGER
from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import DiTCC, DiT
from models.weighting import PerceptionAwareMultimodalAdaptor


class ACDiTRunner(
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
        mobility_head_ckpt_path=None,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super(ACDiTRunner, self).__init__()
        # Create diffusion model
        hidden_size = config["rdt"]["hidden_size"]
        self.model = DiTCC(  
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config["rdt"]["depth"],
            num_heads=config["rdt"]["num_heads"],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            pc_cond_len=pc_cond_len,
            num_latent_feature_tokens=5,
            dtype=dtype,
        )
        self.lift3d = Lift3dSigLip()
        self.lift3d.to(dtype=dtype)  # ! make it bf16

        self.perception_aware_multimodal_adaptor = PerceptionAwareMultimodalAdaptor()

        ##########* [BEGIN] Lightweight Mobility Action Head [BEGIN] *###########
        def extract_model_ckpt(state_dict):
            extracted_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    extracted_state_dict[key[6:]] = value
            return extracted_state_dict
        self.light_weight_mobility_action_head = DiT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=1024,
            depth=14,
            num_heads=config["rdt"]["num_heads"],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            pc_cond_len=pc_cond_len,
            dtype=dtype,
        )
        ckpt_path = pathlib.Path(mobility_head_ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = extract_model_ckpt(ckpt['module'])
        self.light_weight_mobility_action_head.load_state_dict(ckpt)
        self.light_weight_mobility_action_head.to(dtype=dtype)

        for param in self.light_weight_mobility_action_head.parameters():
            param.requires_grad = False

        ############* [END] Lightweight Mobility Action Head [END] *###########

        ##########* [BEGIN] AC-DiT Adapters [BEGIN] *###########
        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config["lang_adaptor"], in_features=lang_token_dim, out_features=hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config["img_adaptor"], in_features=img_token_dim, out_features=hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config["state_adaptor"],
            in_features=state_token_dim * 2,  # state + state mask (indicator)
            out_features=hidden_size,
        )
        # * add lift3d adaptor
        self.lift3d_adaptor = self.build_condition_adapter(
            config["lift3d_adaptor"], in_features=pc_token_dim, out_features=hidden_size
        )
        # * add latent_mobility_feature
        self.latent_mobility_feature = self.build_condition_adapter(
            projector_type="mlp3x_gelu",
            in_features=1024,
            out_features=hidden_size,
        )
        
        # Only create in_context_conditions_adaptor if in_context_cond_dim > 0
        self.in_context_cond_dim = in_context_cond_dim
        if in_context_cond_dim > 0:
            self.in_context_conditions_adaptor = self.build_condition_adapter(
                config["in_context_conditions_adaptor"],  # fallback to goal_pos config
                in_features=in_context_cond_dim,  # 3 (goal_pos) + 1 (is_grasped) + 7 (obj_pose) + 7 (tcp_pose) = 18
                out_features=hidden_size,
            )
        else:
            self.in_context_conditions_adaptor = None
        ############* [END] AC-DiT Adapters [END] *###########

        ##########* [BEGIN] Mobility Head Adapters [BEGIN] *###########
        self.lang_adaptor_mobility_head = self.build_condition_adapter(
            config["lang_adaptor"],
            in_features=lang_token_dim,
            out_features=1024,
        )
        self.img_adaptor_mobility_head = self.build_condition_adapter(
            config["img_adaptor"],
            in_features=img_token_dim,
            out_features=1024,
        )
        self.state_adaptor_mobility_head = self.build_condition_adapter(
            config["state_adaptor"],
            in_features=state_token_dim * 2,
            out_features=1024,
        )
        self.lift3d_adaptor_mobility_head = self.build_condition_adapter(
            config["lift3d_adaptor"],
            in_features=pc_token_dim,
            out_features=1024,
        )
        
        # Only create mobility head in_context_conditions_adaptor if in_context_cond_dim > 0
        if in_context_cond_dim > 0:
            self.in_context_conditions_adaptor_mobility_head = self.build_condition_adapter(
                config["in_context_conditions_adaptor"],  # fallback to goal_pos config
                in_features=18,  # 3 (goal_pos) + 1 (is_grasped) + 7 (obj_pose) + 7 (tcp_pose) = 18
                out_features=1024,
            )
        else:
            self.in_context_conditions_adaptor_mobility_head = None
        ############* [END] Mobility Head Adapters [END] *###########

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
                + [p.numel() for p in self.latent_mobility_feature.parameters()]
                + ([p.numel() for p in self.in_context_conditions_adaptor.parameters()]
                   if self.in_context_conditions_adaptor is not None else [])
            )
        )  # * add lift3d_adaptor, latent_mobility_feature and in_context_conditions_adaptor

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

        # * Load args from config file if not exists in online model config
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
        pc_tokens = pc_tokens.reshape(
            batch_size, T, pc_tokens.shape[-2], pc_tokens.shape[-1]
        )
        pc_tokens = pc_tokens.reshape(
            batch_size, T * pc_tokens.shape[-2], pc_tokens.shape[-1]
        )
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
        latent_mobility_tokens,
        in_context_conditions_tokens, 
    ):
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adpated_pc = self.lift3d_adaptor(pc_tokens)
        adapted_state = self.state_adaptor(state_tokens)
        adapted_latent_mobility_tokens = self.latent_mobility_feature(latent_mobility_tokens)
        
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

        return adapted_lang, adapted_img, adpated_pc, adapted_state, adapted_latent_mobility_tokens, adapted_in_context_conditions

    def adapt_conditions_mobility_head(
        self, 
        lang_tokens, 
        img_tokens, 
        pc_tokens, 
        state_tokens,
        in_context_conditions_tokens, 
    ):
        adapted_lang = self.lang_adaptor_mobility_head(lang_tokens)
        adapted_img = self.img_adaptor_mobility_head(img_tokens)
        adpated_pc = self.lift3d_adaptor_mobility_head(pc_tokens)
        adapted_state = self.state_adaptor_mobility_head(state_tokens)
        
        if self.in_context_conditions_adaptor_mobility_head is not None:
            adapted_in_context_conditions = self.in_context_conditions_adaptor_mobility_head(in_context_conditions_tokens)
        else:
            # Create dummy tensor with correct shape when no in-context conditions
            batch_size = lang_tokens.shape[0]
            device = lang_tokens.device
            dtype = lang_tokens.dtype
            adapted_in_context_conditions = torch.zeros(
                (batch_size, 1, 1024), 
                device=device, 
                dtype=dtype
            )

        return adapted_lang, adapted_img, adpated_pc, adapted_state, adapted_in_context_conditions

    def conditional_sample(
        self,
        lang_cond,
        lang_attn_mask,
        img_cond,
        pc_cond,
        latent_mobility_cond,
        state_traj,
        action_mask,
        ctrl_freqs,
        in_context_conditions_cond,
    ):
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
                lm_c=latent_mobility_cond,
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
    
    def light_weight_mobility_action_head_conditional_sample(
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

        last_tokens = []

        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor_mobility_head(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # Predict the model output
            model_output, last_token = self.light_weight_mobility_action_head.forward_with_last_token(
                x=state_action_traj,
                freq=ctrl_freqs,
                t=t.unsqueeze(-1).to(device),
                lang_c=lang_cond,
                img_c=img_cond,
                pc_c=pc_cond,
                lang_mask=lang_attn_mask,
                # ! unified in-context conditions - pass as single condition
                in_context_conditions_c=in_context_conditions_cond,
            )
            last_tokens.append(last_token)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action
            ).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        last_tokens = torch.stack(last_tokens, dim=1)

        # from termcolor import cprint
        # cprint(last_tokens.shape, 'red')
        # torch.Size([2, 5, 64, 1024])

        return noisy_action, last_tokens

    # ========= Train  ============
    def compute_loss(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        pc_tokens,  # * add pc_tokens
        state_tokens,
        action_gt,
        action_mask,
        ctrl_freqs,
        # ! unified in-context conditions
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

        weighted_img_tokens, weighted_pc_tokens = self.perception_aware_multimodal_adaptor(img_tokens, pc_tokens, lang_tokens)

        with torch.no_grad():
            state_tokens_mobility_head = torch.cat([state_tokens, action_mask], dim=2)
            lang_cond_mobility_head, img_cond_mobility_head, pc_cond_mobility_head, state_traj_mobility_head, \
            in_context_conditions_cond_mobility_head = self.adapt_conditions_mobility_head(
                lang_tokens=lang_tokens,
                img_tokens=img_tokens,
                pc_tokens=pc_tokens,
                state_tokens=state_tokens_mobility_head,
                in_context_conditions_tokens=in_context_conditions,
            )
            mobility_action, last_tokens = self.light_weight_mobility_action_head_conditional_sample(
                lang_cond=lang_cond_mobility_head,
                lang_attn_mask=lang_attn_mask,
                img_cond=img_cond_mobility_head,
                pc_cond=pc_cond_mobility_head,
                state_traj=state_traj_mobility_head,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                in_context_conditions_cond=in_context_conditions_cond_mobility_head,
            )
            last_tokens  # shape: (batch_size, 5, 64, 1024)
            last_tokens = last_tokens.reshape(batch_size, -1, 1024)  # shape: (batch_size, 5 * 64, 1024)
            last_tokens_detached = last_tokens.detach()

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

        lang_cond, img_cond, pc_cond, state_action_traj, latent_mobility_cond, \
        in_context_conditions_cond = self.adapt_conditions(
            lang_tokens=lang_tokens, 
            img_tokens=weighted_img_tokens, 
            pc_tokens=weighted_pc_tokens, 
            state_tokens=state_action_traj, 
            latent_mobility_tokens=last_tokens_detached,
            in_context_conditions_tokens=in_context_conditions,
        )
        # Predict the denoised result
        pred = self.model(
            x=state_action_traj,
            freq=ctrl_freqs,
            t=timesteps,
            lang_c=lang_cond,
            img_c=img_cond,
            lm_c=latent_mobility_cond,
            pc_c=pc_cond,
            lang_mask=lang_attn_mask,
            # ! unified in-context conditions
            in_context_conditions_c=in_context_conditions_cond,
        )

        pred_type = self.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # * combine the loss
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
        # ! unified in-context conditions
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
        batch_size = lang_tokens.shape[0]

        img_tokens, pc_tokens = self.perception_aware_multimodal_adaptor(img_tokens, pc_tokens, lang_tokens)

        with torch.no_grad():
            state_tokens = torch.cat([state_tokens, action_mask], dim=2)
            lang_cond_mobility_head, img_cond_mobility_head, pc_cond_mobility_head, state_traj_mobility_head, \
            in_context_conditions_cond_mobility_head = self.adapt_conditions_mobility_head(
                lang_tokens=lang_tokens,
                img_tokens=img_tokens,
                pc_tokens=pc_tokens,
                state_tokens=state_tokens,
                in_context_conditions_tokens=in_context_conditions,
            )
            mobility_action, hidden_states = self.light_weight_mobility_action_head_conditional_sample(
                lang_cond=lang_cond_mobility_head,
                lang_attn_mask=lang_attn_mask,
                img_cond=img_cond_mobility_head,
                pc_cond=pc_cond_mobility_head,  # * add pc_cond
                state_traj=state_traj_mobility_head,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                in_context_conditions_cond=in_context_conditions_cond_mobility_head,
            )
            hidden_states  # shape: (batch_size, 5, 64, 1024)
            hidden_states = hidden_states.reshape(batch_size, -1, 1024)  # shape: (batch_size, 5 * 64, 1024)
            hidden_states_detached = hidden_states.detach()

        # Prepare the state and conditions
        lang_cond, img_cond, pc_cond, state_traj, latent_mobility_cond, \
        in_context_conditions_cond = self.adapt_conditions(
            lang_tokens=lang_tokens,
            img_tokens=img_tokens,
            pc_tokens=pc_tokens,
            state_tokens=state_tokens,
            latent_mobility_tokens=hidden_states_detached,
            in_context_conditions_tokens=in_context_conditions,
        )

        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond=lang_cond,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            pc_cond=pc_cond, 
            latent_mobility_cond=latent_mobility_cond, 
            state_traj=state_traj,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs,
            in_context_conditions_cond=in_context_conditions_cond,
        )

        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
