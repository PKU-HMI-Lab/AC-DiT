import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.acdit_runner import ACDiTRunner
from models.dit_runner import DiTRunner

import sys
import pathlib
project_root = pathlib.Path(__file__).parents[1]
sys.path.append(str(project_root))
from helpers.common import ACDIT_DEFAULT_LOGGER


MSHAB_ACTION_INDICES = [
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



def create_model(args, pretrained, **kwargs):
    model = MSHabModel(args, **kwargs)
    if pretrained is not None:
        model.load_pretrained_weights(pretrained)
    return model


class MSHabModel(object):
    def __init__(
        self, args,
        method_name='AC-DiT',
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=20,
        pretrained_vision_encoder_name_or_path=None,
        combine_flag=False,
        mobility_head_ckpt_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.method_name = method_name
        self.device = device
        self.control_frequency = control_frequency
        self.combine_flag = combine_flag
        self.mobility_head_ckpt_path = mobility_head_ckpt_path
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy()

        self.reset()

    def get_policy(self):
        img_cond_len = (self.args["common"]["img_history_size"]
                        * self.args["common"]["num_cameras"]
                        * self.vision_model.num_patches)

        pc_cond_len = self.args["common"]["img_history_size"] * self.args["dataset"]["pointcloud_feature_length"]

        if self.method_name == "AC-DiT":
            RUNNER_CLS = ACDiTRunner
        elif self.method_name == "DiT":
            RUNNER_CLS = DiTRunner
        else:
            raise ValueError(f'Unknown method name {self.method_name}')

        init_kwargs = {
            "action_dim": self.args["common"]["state_dim"],
            "pred_horizon": self.args["common"]["action_chunk_size"],
            "config": self.args["model"],
            "lang_token_dim": self.args["model"]["lang_token_dim"],
            "img_token_dim": self.args["model"]["img_token_dim"],
            "pc_token_dim": self.args["model"]["pc_token_dim"],
            "state_token_dim": self.args["model"]["state_token_dim"],
            "max_lang_cond_len": self.args["dataset"]["tokenizer_max_length"],
            "img_cond_len": img_cond_len,
            "pc_cond_len": pc_cond_len,
            "in_context_cond_dim": 18,
            "dtype": self.dtype,
        }

        # Only pass mobility_head_ckpt_path for AC-DiT
        if self.method_name == "AC-DiT":
            init_kwargs["mobility_head_ckpt_path"] = self.mobility_head_ckpt_path

        _model = RUNNER_CLS(**init_kwargs)

        return _model

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint =  torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def random_set_language(self, dataset_dir):
        instructions_dir = pathlib.Path(dataset_dir) / "instructions"
        instructions_files = list(instructions_dir.glob("*.pt"))
        if len(instructions_files) == 0:
            ACDIT_DEFAULT_LOGGER.error(f"No instruction embeddings found in {instructions_dir}")
            exit(-1)
        choice = np.random.choice(len(instructions_files))
        self.lang_embeddings = torch.load(instructions_files[choice])["text_embed"]
        ACDIT_DEFAULT_LOGGER.info(f"Loaded language embeddings from {instructions_files[choice]}")

    def _format_joint_to_state(self, joints):
        """
        Format the robot joint state into the unified state vector.

        Args:
            joints (torch.Tensor): The joint state to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted state ([B, N, 128]). 
        """
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]), 
            device=joints.device, dtype=joints.dtype
        )
        # assemble the unifed state vector
        state[:, :, MSHAB_ACTION_INDICES] = joints # state -> unified
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        ) 
        state_elem_mask[:, MSHAB_ACTION_INDICES] = 1 # action mask
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action): # unified -> action
        action_indices = MSHAB_ACTION_INDICES
        joints = action[:, :, action_indices]
        return joints

    @torch.no_grad()
    def step(self, proprio, images, text_embeds, pointclouds, extra_obs):
        """
        Args:
            proprio: proprioceptive states
            images: RGB images
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype
        
        # background image
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        # preprocess images:
        # 1. replace void images with the background image
        # 2. resize images to `self.data_args.image_size`
        # 3. adjuest brightness: if average_brightness <= 0.15 then increase it to 1.75 times
        # 4. pad images to squares
        # 5. use `SiglipImageProcessor` to process images
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
            
            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)
            
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)  
        image_embeds_uncat = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds_uncat.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # history of actions
        joints = proprio.to(device).unsqueeze(0)
        states, state_elem_mask = self._format_joint_to_state(joints)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)

        text_embeds = text_embeds.to(device, dtype=dtype)
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)

        goal_pos = extra_obs["goal_pos_wrt_base"].to(device, dtype=dtype)
        is_grasped = extra_obs["is_grasped"].to(device, dtype=dtype)
        obj_pose = extra_obs["obj_pose_wrt_base"].to(device, dtype=dtype)
        tcp_pose = extra_obs["tcp_pose_wrt_base"].to(device, dtype=dtype)
        
        in_context_conditions = torch.cat([
            goal_pos,           # 3 dimensions: [x, y, z]
            is_grasped.unsqueeze(0),         # 1 dimension: [grasped_flag]
            obj_pose,           # 7 dimensions: [x, y, z, qx, qy, qz, qw]
            tcp_pose            # 7 dimensions: [x, y, z, qx, qy, qz, qw]
        ], dim=-1)  # Add batch dimension: [1, 18]

        if pointclouds[0] is None:
            pointclouds[0] = pointclouds[1]
        pointcloud_tensor_arrs = [torch.from_numpy(arr).to(device, dtype=dtype) for arr in pointclouds]
        pointcloud = torch.stack(pointcloud_tensor_arrs, dim=0).unsqueeze(0)
        pc_tokens = self.policy.encode_pointcloud(pointcloud, weight_dtype=torch.bfloat16)

        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            pc_tokens=pc_tokens,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs,
            in_context_conditions=in_context_conditions,
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory
