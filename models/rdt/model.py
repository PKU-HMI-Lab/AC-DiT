from collections import OrderedDict

import torch
import torch.nn as nn

from helpers.common import ACDIT_DEFAULT_LOGGER
from models.rdt.blocks import (
    FinalLayer,
    RDTBlock,
    TimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
    get_multimodal_cond_pos_embed,
)


class DiT(nn.Module):
    """
    Multi-modal Robotics Diffusion Transformer (5D version).
    
    This model processes multi-modal inputs (language, images, point clouds, and unified 
    in-context conditions) to generate robot action sequences using diffusion-based 
    generation. The "5D" refers to the five main condition modalities:
    1. Language instructions
    2. Visual observations (images)
    3. Point cloud data
    4. Unified in-context conditions (goal position, grasp state, object/TCP poses)
    5. Temporal context (timestep and control frequency)
    
    The model uses a transformer architecture with cross-attention between the main
    sequence (state + action tokens) and the various condition modalities.
    """
    
    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        pc_cond_len=1024,  # Point cloud condition length
        dtype=torch.bfloat16,
    ):
        """
        Initialize the DiT model.
        
        Args:
            output_dim (int): Dimension of the output action space
            horizon (int): Number of action steps to predict
            hidden_size (int): Hidden dimension of the transformer
            depth (int): Number of transformer layers
            num_heads (int): Number of attention heads
            max_lang_cond_len (int): Maximum length of language condition tokens
            img_cond_len (int): Fixed length of image condition tokens
            pc_cond_len (int): Length of point cloud condition tokens
            dtype (torch.dtype): Data type for model parameters
        """
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.pc_cond_len = pc_cond_len
        self.dtype = dtype

        # Timestep and control frequency embedders for temporal conditioning
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # Learnable position embeddings for different modalities
        # Main sequence: [timestep; control_freq; in_context_conditions; state; action_sequence]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, 67, hidden_size))
        
        # Transformer blocks for processing sequences with cross-attention
        self.blocks = nn.ModuleList(
            [RDTBlock(hidden_size, num_heads) for _ in range(depth)]
        )
        
        # Final layer to project to output action dimensions
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize model weights using appropriate initialization strategies.
        
        This method initializes:
        1. Transformer layer weights (Xavier uniform)
        2. Position embeddings (sin-cos embeddings)
        3. Timestep and frequency embedders (normal distribution)
        4. Final layer weights (zero initialization for stable training)
        5. Converts all parameters to the specified dtype
        """
        # Initialize transformer layers with Xavier uniform initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize main sequence position embeddings using sin-cos embeddings
        # Sequence structure: [timestep, ctrl_freq, in_context_conditions, state, action_horizon, redundant_padding]
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict(
                [
                    ("timestep", 1),           # Diffusion timestep
                    ("ctrl_freq", 1),          # Control frequency
                    ("in_context_conditions", 1),  # Unified in-context conditions
                    ("state", 1),              # Current state
                    ("action", self.horizon),  # Action sequence
                    ("redundant_padding", 61), # Padding for alignment
                ]
            ),
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # Initialize timestep and control frequency embedding MLPs
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        # Initialize final layer with zero weights for stable training
        # This helps with diffusion training stability
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

        # Convert all parameters to the specified data type
        self.to(self.dtype)

    def forward(
        self,
        x,
        freq,
        in_context_conditions_c,
        t,
        lang_c,
        img_c,
        pc_c,
        lang_mask=None,
        img_mask=None,
        pc_mask=None,
    ):
        """
        Forward pass of the DiT model.
        
        This method processes a sequence of state and action tokens along with 
        multi-modal conditions to generate refined action predictions using 
        diffusion-based denoising.

        Args:
            x (torch.Tensor): State + action token sequence of shape (B, T, D)
                where T = horizon + 1 (includes current state + action sequence)
            freq (torch.Tensor): Control frequency scalar of shape (B,)
            in_context_conditions_c (torch.Tensor): Unified in-context conditions
                of shape (B, 1, D). This tensor contains the concatenated and 
                pre-adapted goal_pos, is_grasped, obj_pose, and tcp_pose information.
            t (torch.Tensor): Diffusion timesteps of shape (B,) or (1,)
            lang_c (torch.Tensor): Language condition tokens of shape (B, L_lang, D)
            img_c (torch.Tensor): Image condition tokens of shape (B, L_img, D)
            pc_c (torch.Tensor): Point cloud condition tokens of shape (B, L_pc, D)
            lang_mask (torch.Tensor, optional): Language condition mask of shape (B, L_lang)
                where True indicates valid tokens
            img_mask (torch.Tensor, optional): Image condition mask of shape (B, L_img)
                where True indicates valid tokens
            pc_mask (torch.Tensor, optional): Point cloud condition mask of shape (B, L_pc)
                where True indicates valid tokens

        Returns:
            torch.Tensor: Predicted action sequence of shape (B, horizon, output_dim)
        """
        # Embed temporal information
        t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # Handle in_context_conditions_c - it may be None for some models
        if in_context_conditions_c is not None:
            in_context_conditions_c = in_context_conditions_c.unsqueeze(1)  # (B, 1, D)
        else:
            # Create dummy zero tensor when no in-context conditions
            batch_size = x.shape[0]
            device = x.device
            dtype = x.dtype
            in_context_conditions_c = torch.zeros(
                (batch_size, 1, self.hidden_size), 
                device=device, 
                dtype=dtype
            )

        # Expand timestep embedding to match batch size if needed
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        
        # Construct main sequence: [timestep, freq, in_context_conditions, state+actions]
        # Note: in_context_conditions_c is already adapted to hidden_size dimension
        x = torch.cat([t, freq, in_context_conditions_c, x], dim=1) 
        
        # Add positional embeddings to main sequence
        # We use the first 6 positions: [timestep, freq, in_context_conditions, state, action_start, ...]
        x = x + self.x_pos_embed[:,:6,:]

        # Concatenate all condition modalities
        conditions = torch.cat([lang_c, img_c, pc_c], dim=1)
        
        # Create attention masks for conditions
        cond_masks = [
            lang_mask if lang_mask is not None else torch.ones_like(lang_c[:, :, 0], dtype=torch.bool),
            img_mask if img_mask is not None else torch.ones_like(img_c[:, :, 0], dtype=torch.bool),
            pc_mask if pc_mask is not None else torch.ones_like(pc_c[:, :, 0], dtype=torch.bool),
        ]
        cond_mask = torch.cat(cond_masks, dim=1)  # (B, L_cond_total)

        # Process through transformer blocks with cross-attention
        for block in self.blocks:
            x = block(x, conditions, cond_mask)
            
        # Apply final layer to get action predictions
        x = self.final_layer(x)  # (B, T+1, out_channels)

        # Extract only the action tokens (exclude timestep, freq, in_context_conditions, state)
        x = x[:, -self.horizon :]
        return x
    
    def forward_with_last_token(
        self,
        x,
        freq,
        t,
        lang_c,
        img_c,
        pc_c,
        in_context_conditions_c,
        lang_mask=None,
        img_mask=None,
        pc_mask=None,
    ):
        """
        Forward pass with last token extraction for analysis purposes.
        
        This method is similar to the standard forward pass but additionally
        returns the hidden states from the last transformer layer before
        the final projection. This is useful for analysis, visualization,
        or downstream tasks that require access to the learned representations.

        Args:
            x (torch.Tensor): State + action token sequence of shape (B, T, D)
            freq (torch.Tensor): Control frequency scalar of shape (B,)
            t (torch.Tensor): Diffusion timesteps of shape (B,) or (1,)
            lang_c (torch.Tensor): Language condition tokens of shape (B, L_lang, D)
            img_c (torch.Tensor): Image condition tokens of shape (B, L_img, D)
            pc_c (torch.Tensor): Point cloud condition tokens of shape (B, L_pc, D)
            in_context_conditions_c (torch.Tensor): Unified in-context conditions
                of shape (B, 1, D). Contains pre-adapted goal_pos, is_grasped, obj_pose, tcp_pose
            lang_mask (torch.Tensor, optional): Language condition mask of shape (B, L_lang)
            img_mask (torch.Tensor, optional): Image condition mask of shape (B, L_img)
            pc_mask (torch.Tensor, optional): Point cloud condition mask of shape (B, L_pc)
            
        Returns:
            tuple: (predictions, last_token) where:
                - predictions (torch.Tensor): Action predictions of shape (B, horizon, output_dim)
                - last_token (torch.Tensor): Hidden states of shape (B, horizon, hidden_size)
        """
        # Embed temporal information
        t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        if in_context_conditions_c.dim() == 2:
            # If in_context_conditions_c is 2D, expand to 3D
            in_context_conditions_c = in_context_conditions_c.unsqueeze(1)

        # Expand timestep embedding to match batch size if needed
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        
        # Construct main sequence with unified in-context conditions
        x = torch.cat([t, freq, in_context_conditions_c, x], dim=1) 

        # Add positional embeddings with dynamic slicing
        # This allows for flexible sequence lengths while maintaining positional awareness
        x = x + self.x_pos_embed[:, :x.shape[1], :]

        # Concatenate all condition modalities
        conditions = torch.cat([lang_c, img_c, pc_c], dim=1)
        
        # Create attention masks for conditions
        cond_masks = [
            lang_mask if lang_mask is not None else torch.ones_like(lang_c[:, :, 0], dtype=torch.bool),
            img_mask if img_mask is not None else torch.ones_like(img_c[:, :, 0], dtype=torch.bool),
            pc_mask if pc_mask is not None else torch.ones_like(pc_c[:, :, 0], dtype=torch.bool),
        ]
        cond_mask = torch.cat(cond_masks, dim=1)  # (B, L_cond_total)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, conditions, cond_mask)

        # Extract last token representations before final projection
        last_token = x.clone()
        last_token = last_token[:, -self.horizon :]
            
        # Apply final layer to get action predictions
        x = self.final_layer(x)  # (B, T+1, out_channels)

        # Extract only the action tokens
        x = x[:, -self.horizon :]
        return x, last_token


class DiTCC(nn.Module):
    """
    Multi-modal Robotics Diffusion Transformer with Cyclic Conditioning.
    
    This is an extended version of DiTCC that includes support for "guess actions" 
    (cyclic conditioning). The model can incorporate previous action predictions or
    action priors as additional conditioning information, which helps with:
    1. Iterative refinement of action predictions
    2. Consistency across multiple forward passes
    3. Better handling of temporal dependencies
    
    The "CC" (Cyclic Conditioning) refers to the ability to use previous predictions
    as conditions for subsequent predictions, creating a feedback loop that can
    improve action quality over multiple iterations.
    """

    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        pc_cond_len=1024,  # Point cloud condition length
        num_latent_feature_tokens=1,
        dtype=torch.bfloat16,
    ):
        """
        Initialize the DiTCC model.
        
        Args:
            output_dim (int): Dimension of the output action space
            horizon (int): Number of action steps to predict
            hidden_size (int): Hidden dimension of the transformer
            depth (int): Number of transformer layers
            num_heads (int): Number of attention heads
            max_lang_cond_len (int): Maximum length of language condition tokens
            img_cond_len (int): Fixed length of image condition tokens
            pc_cond_len (int): Length of point cloud condition tokens
            num_latent_feature_tokens (int): Number of guess action tokens per timestep
            dtype (torch.dtype): Data type for model parameters
        """
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.pc_cond_len = pc_cond_len
        self.num_latent_feature_tokens = num_latent_feature_tokens
        self.dtype = dtype

        # Timestep and control frequency embedders
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # Learnable position embeddings for different modalities
        # Main sequence: [timestep; control_freq; in_context_conditions; state; action_sequence]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, 67, hidden_size))
        
        # Guess action condition position embeddings
        # This allows the model to condition on previous action predictions
        self.guess_action_cond_pos_embed = nn.Parameter(
            torch.zeros(1, horizon * self.num_latent_feature_tokens, hidden_size)
        )

        # Transformer blocks for processing sequences with cross-attention
        self.blocks = nn.ModuleList(
            [RDTBlock(hidden_size, num_heads) for _ in range(depth)]
        )

        # Final layer to project to output action dimensions
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize model weights using appropriate initialization strategies.
        
        This method initializes all model components similar to DiTCC but with
        additional initialization for guess action position embeddings.
        """
        # Initialize transformer layers with Xavier uniform initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize main sequence position embeddings
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict(
                [
                    ("timestep", 1),
                    ("ctrl_freq", 1),
                    ("in_context_conditions", 1),
                    ("state", 1),
                    ("action", self.horizon),
                    ("redundant_padding", 61), # Padding for alignment
                ]
            ),
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # Initialize guess action condition position embeddings
        # This uses simple 1D sin-cos embeddings for the guess action sequence
        guess_action_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.hidden_size, torch.arange(self.horizon * self.num_latent_feature_tokens)
        )
        self.guess_action_cond_pos_embed.data.copy_(
            torch.from_numpy(guess_action_cond_pos_embed).float().unsqueeze(0)
        )

        # Initialize timestep and control frequency embedding MLPs
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        # Initialize final layer with zero weights for stable training
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

        # Convert all parameters to the specified data type
        self.to(self.dtype)

    def forward(
        self,
        x,
        freq,
        t,
        lang_c,
        img_c,
        pc_c,
        lm_c,
        # ! unified in-context conditions
        in_context_conditions_c,
        lang_mask=None,
        img_mask=None,
        pc_mask=None,
        lm_mask=None,
    ):
        """
        Forward pass of DiTCC with unified in-context conditions.

        Args:
            x: (B, T, D) - State + action token sequence, T = horizon + 1
            freq: (B,) - Control frequency scalar
            t: (B,) or (1,) - Diffusion timesteps
            lang_c: (B, L_lang, D) - Language condition tokens (variable length)
            img_c: (B, L_img, D) - Image condition tokens (fixed length)
            pc_c: (B, L_pc, D) - Point cloud condition tokens
            lm_c: (B, horizon, D) - Guess action condition tokens
            in_context_conditions_c: (B, 1, D) - Unified in-context conditions (adapted)
                This contains the pre-adapted concatenated goal_pos, is_grasped, obj_pose, tcp_pose
            lang_mask: (B, L_lang) - Language condition mask (True for valid)
            img_mask: (B, L_img) - Image condition mask (True for valid)
            pc_mask: (B, L_pc) - Point cloud condition mask (True for valid)
            lm_mask: (B, horizon) - Guess action condition mask (True for valid)
        
        Returns:
            torch.Tensor: (B, horizon, output_dim) - Predicted action sequence
        """
        t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        if in_context_conditions_c.dim() == 2:
            # If in_context_conditions_c is 2D, expand to 3D
            in_context_conditions_c = in_context_conditions_c.unsqueeze(1)

        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        
        # Use unified in-context conditions instead of separate conditions
        # in_context_conditions_c is already adapted to hidden_size dimension
        x = torch.cat([t, freq, in_context_conditions_c, x], dim=1) 

        # Add multimodal position embeddings
        # Adjust position embedding indexing since we now have fewer tokens in the sequence
        x = x + self.x_pos_embed[:, :6, :]  # Dynamic slicing based on actual sequence length
        lm_c = lm_c + self.guess_action_cond_pos_embed  # * add guess action condition
        
        # concate all conditions
        conditions = torch.cat([lang_c, img_c, pc_c, lm_c], dim=1)
        cond_masks = [
            lang_mask if lang_mask is not None else torch.ones_like(lang_c[:, :, 0], dtype=torch.bool),
            img_mask if img_mask is not None else torch.ones_like(img_c[:, :, 0], dtype=torch.bool),
            pc_mask if pc_mask is not None else torch.ones_like(pc_c[:, :, 0], dtype=torch.bool),
            lm_mask if lm_mask is not None else torch.ones_like(lm_c[:, :, 0], dtype=torch.bool),
        ]
        cond_mask = torch.cat(cond_masks, dim=1)  # (B, L_cond_total)

        # Forward pass
        for block in self.blocks:
            x = block(x, conditions, cond_mask)
            
        # final layer
        x = self.final_layer(x)  # (B, T+1, out_channels)

        # Only preserve the action tokens
        x = x[:, -self.horizon :]
        return x
