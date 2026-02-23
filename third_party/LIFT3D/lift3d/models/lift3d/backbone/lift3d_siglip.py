from peft import get_peft_model, LoraConfig
import torch
from torch import nn
from transformers import AutoModelForImageClassification, AutoTokenizer
from transformers.models.siglip.modeling_siglip import SiglipModel
from lift3d.helpers.pytorch import log_params_to_file
from lift3d.helpers.common import Logger
from lift3d.helpers.graphics import PointCloud
import os
import torch.nn.functional as F

from lift3d.models.lift3d.model_utils.mv_utils import PCViews
from lift3d.models.lift3d.model_utils.networks import Point_PN_scan


class Lift3dSigLip(nn.Module):
    """SiGLIP model modified for 3D point cloud processing.
    
    This model extends the original SiGLIP architecture by:
    1. Adding LoRA adapters to attention layers
    2. Converting 3D point clouds to 2D token representations
    3. Maintaining positional embeddings through bilinear interpolation
    """
    def __init__(self, config=None, **kwargs):
        super(Lift3dSigLip, self).__init__()
        
        # Initialize base SiGLIP model
        model_name = "google/siglip-so400m-patch14-384"
        base_model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Initialize LoRA configuration
        lora_config = LoraConfig(
            r=2,
            lora_alpha=1,
            lora_dropout=0.25,
            target_modules=["q_proj", "k_proj", "v_proj"],
            inference_mode=False,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )

        # Apply LoRA to vision encoder
        self.vision_model = get_peft_model(base_model.vision_model, lora_config)
        self.model = base_model

        # Initialize Lift3D specific configurations
        self.trans_dim = 1152
        self.patchknn = 64
        self.layer_norm_eps = 1e-06
        self.patch_embed = Point_PN_scan(k_neighbors=self.patchknn)
        
        # Initialize learnable tokens and embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        
        # Initialize point cloud view generator
        pc_views = PCViews()
        self.get_pos_2d = pc_views.get_pos_2d

        # Register 2D positional embeddings
        weight = self.model.vision_model.embeddings.position_embedding.weight
        weight = weight.unsqueeze(0)
        self.register_parameter('pos_embed_2d', nn.Parameter(weight, requires_grad=False))

        # Initialize dimension transformation layers
        self.token_dim_transform = nn.Linear(768, 1152)
        self.norm = nn.LayerNorm(self.trans_dim, eps=self.layer_norm_eps)
        self.feature_dim = self.trans_dim

    def bilinear_interpolation_3d_to_2d(self, x, y, pos_embed):
        """Perform bilinear interpolation from 3D coordinates to 2D position embeddings."""
        img_size = 384
        patch_size = 14
        
        grid_x = (2.0 * x / (img_size-1)) - 1
        grid_y = (2.0 * y / (img_size-1)) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(2)

        pos_embed_reshaped = pos_embed.permute(0, 2, 1).view(
            1, -1, int(img_size / patch_size), int(img_size / patch_size)
        ).repeat(grid.shape[0], 1, 1, 1)
        pos_embed_reshaped = pos_embed_reshaped.cuda()
        
        return F.grid_sample(
            pos_embed_reshaped, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        ).squeeze()

    def forward(self, pts):
        """Forward pass for processing point clouds.
        
        Args:
            pts (Tensor): Input point cloud tensor of shape [B, N, 3] or [N, 3]
        Returns:
            Tensor: Extracted features of shape [B, feature_dim]
        """
        batch_size = pts.shape[0]
        tokens, pos = [], []

        if len(pts.shape) == 2:
            pts = pts.unsqueeze(0).float()

        pts = PointCloud.normalize(pts)
        pts = pts[:, :, :3]
        
        pts_trans = pts.clone().transpose(1, 2).contiguous()
        center, group_input_tokens = self.patch_embed(pts_trans, pts)
        group_input_tokens = group_input_tokens.transpose(1, 2)

        pos_x, pos_y, _ = self.get_pos_2d(center)
        pos_x = pos_x.to(dtype=pts.dtype)
        pos_y = pos_y.to(dtype=pts.dtype)
        interpolated_pos_embed = self.bilinear_interpolation_3d_to_2d(
            pos_x, pos_y, self.pos_embed_2d.to(dtype=pts.dtype)
        )
        interpolated_pos_embed = interpolated_pos_embed.reshape(
            center.shape[0], -1, center.shape[1], self.trans_dim
        )
        interpolated_pos_embed = interpolated_pos_embed.mean(dim=1)

        group_input_tokens = self.token_dim_transform(group_input_tokens)
        tokens.append(group_input_tokens)
        pos.append(interpolated_pos_embed)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        tokens.insert(0, cls_tokens)
        pos.insert(0, cls_pos)

        pos = torch.cat(pos, dim=1)
        tokens = torch.cat(tokens, dim=1)
        x = tokens + pos

        x = self.model.vision_model.encoder(inputs_embeds=x)
        x = x.last_hidden_state
        
        x = self.norm(x)
        x = x[:, 1:, :]

        return x


def test_image_forward(model_name="google/siglip-so400m-patch14-384"):
    """Test the original SiGLIP model's forward pass with 2D images.
    
    Args:
        model_name (str): Name or path of the pretrained SiGLIP model
    Returns:
        torch.Tensor: Image features of shape [batch_size, hidden_size]
    """
    model = AutoModelForImageClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # SiGLIP expects 384x384 input images
    img = torch.randn(4, 3, 384, 384).to(device)
    
    with torch.amp.autocast(device_type="cuda"):
        outputs = model.vision_model(img)
        image_features = outputs.pooler_output
        
    print("Input image shape:", img.shape)
    print("Output features shape:", image_features.shape)
    print("Hidden size:", model.config.hidden_size)
    
    return image_features


if __name__ == "__main__":
    model_name = "google/siglip-so400m-patch14-384"
    
    print("\nInitializing Lift3dSigLip model...")
    model = Lift3dSigLip()
    
    # Log trainable and frozen parameters
    log_dir = "/home/secure/Codes/ForkedProjects/LIFT3D_CSCSX/outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_params_to_file(model, os.path.join(log_dir, "trainable_params.txt"), requires_grad=True)
    log_params_to_file(model, os.path.join(log_dir, "freeze_params.txt"), requires_grad=False)
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pts = torch.randn(4, 1024, 3).to(device)
    
    with torch.amp.autocast(device_type="cuda"):
        embedding = model(pts)
    
    print("\nForward pass test:")
    print("Input point cloud shape:", pts.shape)
    print("Output embedding shape:", embedding.shape)
