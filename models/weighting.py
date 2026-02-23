import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionAwareMultimodalAdaptor(nn.Module):
    def __init__(self, 
                 image_feat_dim=1152,
                 pc_feat_dim=1152,
                 lang_feat_dim=1152,
                 proj_dim=256):
        super().__init__()
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ).to(dtype=torch.bfloat16)
        
        self.pc_proj = nn.Sequential(
            nn.Linear(pc_feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ).to(dtype=torch.bfloat16)
        
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ).to(dtype=torch.bfloat16)
        
        self.proj_dim = proj_dim
        
    def forward(self, image_feature, pc_feature, lang_tokens):
        weighted_images, weighted_pcs, image_weights, pc_weights = self.forward_with_weights(image_feature, pc_feature, lang_tokens)
        return weighted_images, weighted_pcs
    
    def forward_with_weights(self, image_feature, pc_feature, lang_tokens):
        """
        Args:
            image_feature: [batch_size , 6 * 729, 1152] (6 timesteps per batch)
            pc_feature: [batch_size, 2 * 128, 768] (2 timesteps per batch)
            lang_tokens: [batch_size, length, 4096] (shared across timesteps)
        Returns:
            weighted_image_feature: [batch_size , 6 * 729, 1152]
            weighted_pc_feature: [batch_size, 2 * 128, 768]
        """
        orig_image_dtype = image_feature.dtype
        orig_pc_dtype = pc_feature.dtype
        orig_lang_dtype = lang_tokens.dtype
        
        batch_size = lang_tokens.shape[0]  # 24
        
        image_feature = image_feature.to(dtype=torch.bfloat16)
        pc_feature = pc_feature.to(dtype=torch.bfloat16)
        lang_tokens = lang_tokens.to(dtype=torch.bfloat16)
        
        image_per_batch = image_feature.reshape(batch_size, 6, 729, 1152)
        pc_per_batch = pc_feature.reshape(batch_size, 2, 128, 1152)
        lang_per_batch = lang_tokens.unsqueeze(1) 

        # Compute global features
        lang_global = lang_per_batch.mean(dim=2)  # [24, 1, 4096]
        lang_proj = self.lang_proj(lang_global)  # [24, 1, proj_dim]

        image_globals = image_per_batch.mean(dim=2)  # [24, 6, 1152]
        image_projs = self.image_proj(image_globals)  # [24, 6, proj_dim]

        pc_globals = pc_per_batch.mean(dim=2)  # [24, 2, 1152]
        pc_projs = self.pc_proj(pc_globals)  # [24, 2, proj_dim]

        # Concatenate projections
        all_obs_projs = torch.cat([image_projs, pc_projs], dim=1)

        # Compute similarities
        lang_proj_expanded = lang_proj.expand(-1, all_obs_projs.size(1), -1)

        sims = F.cosine_similarity(all_obs_projs, lang_proj_expanded, dim=2)
        weights = sims / 2 + 1  # [-1, 1] --> [0.5, 1.5]

        image_weights = weights[:, :6].reshape(batch_size, 6, 1, 1).to(orig_image_dtype)
        pc_weights = weights[:, 6:].reshape(batch_size, 2, 1, 1).to(orig_pc_dtype)

        # Apply weights
        weighted_images = (image_per_batch * image_weights).reshape(batch_size, 6 * 729, 1152)
        weighted_pcs = (pc_per_batch * pc_weights).reshape(batch_size, 2 * 128, 1152)

        return weighted_images, weighted_pcs, image_weights, pc_weights
    

if __name__ == "__main__":
    # Example usage
    image_feature = torch.randn(2, 6 * 729, 1152)
    pc_feature = torch.randn(2, 2 * 128, 1152)
    lang_tokens = torch.randn(2, 24, 4096)

    feature_weighting = PerceptionAwareMultimodalAdaptor()
    weighted_images, weighted_pcs = feature_weighting(image_feature, pc_feature, lang_tokens)
    
    print(weighted_images.shape)  # Should be [2, 6 * 729, 1152]
    print(weighted_pcs.shape)     # Should be [2, 2 * 128, 768]
