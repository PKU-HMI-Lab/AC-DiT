from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn

from mshab.agents.sac.modules import Actor, Critic, Encoder, RLProjection, SharedCNN


class Agent(nn.Module):
    def __init__(
        self,
        pixels_obs_space: spaces.Dict,
        state_obs_shape,
        action_shape,
        actor_hidden_dims,
        critic_hidden_dims,
        critic_layer_norm,
        critic_dropout,
        encoder_pixels_feature_dim,
        encoder_state_feature_dim,
        cnn_features,
        cnn_filters,
        cnn_strides,
        cnn_padding,
        log_std_min,
        log_std_max,
        device,
    ):
        super().__init__()

        state_obs_dim = np.prod(state_obs_shape)
        action_dim = np.prod(action_shape)

        shared_cnns: nn.ModuleDict = nn.ModuleDict(
            (
                k,
                SharedCNN(
                    pixel_obs_shape=space.shape,
                    features=cnn_features,
                    filters=cnn_filters,
                    strides=cnn_strides,
                    padding=cnn_padding,
                ),
            )
            for k, space in pixels_obs_space.items()
        )

        target_cnns: nn.ModuleDict = nn.ModuleDict(
            (
                k,
                SharedCNN(
                    pixel_obs_shape=space.shape,
                    features=cnn_features,
                    filters=cnn_filters,
                    strides=cnn_strides,
                    padding=cnn_padding,
                ),
            )
            for k, space in pixels_obs_space.items()
        )

        actor_encoder = Encoder(
            cnns=shared_cnns,
            pixels_projections=nn.ModuleDict(
                (
                    k,
                    RLProjection(
                        cnn.out_dim,
                        encoder_pixels_feature_dim,
                    ),
                )
                for k, cnn in shared_cnns.items()
            ),
            state_projection=RLProjection(
                state_obs_dim,
                encoder_state_feature_dim,
            ),
        )

        critic_encoder = Encoder(
            cnns=shared_cnns,
            pixels_projections=nn.ModuleDict(
                (
                    k,
                    RLProjection(
                        cnn.out_dim,
                        encoder_pixels_feature_dim,
                    ),
                )
                for k, cnn in shared_cnns.items()
            ),
            state_projection=RLProjection(
                state_obs_dim,
                encoder_state_feature_dim,
            ),
        )

        critic_encoder_target = Encoder(
            cnns=target_cnns,
            pixels_projections=nn.ModuleDict(
                (
                    k,
                    RLProjection(
                        cnn.out_dim,
                        encoder_pixels_feature_dim,
                    ),
                )
                for k, cnn in target_cnns.items()
            ),
            state_projection=RLProjection(
                state_obs_dim,
                encoder_state_feature_dim,
            ),
        )

        self.actor = Actor(
            encoder=actor_encoder,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(device)

        self.critic = Critic(
            encoder=critic_encoder,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            layer_norm=critic_layer_norm,
            dropout=critic_dropout,
        ).to(device)

        self.critic_target = Critic(
            encoder=critic_encoder_target,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            layer_norm=critic_layer_norm,
            dropout=critic_dropout,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update_params(self, critic_tau, encoder_tau):
        for param, target_param in zip(
            self.critic.Q1.parameters(), self.critic_target.Q1.parameters()
        ):
            target_param.data.copy_(
                critic_tau * param.data + (1 - critic_tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic.Q2.parameters(), self.critic_target.Q2.parameters()
        ):
            target_param.data.copy_(
                critic_tau * param.data + (1 - critic_tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic.encoder.parameters(), self.critic_target.encoder.parameters()
        ):
            target_param.data.copy_(
                encoder_tau * param.data + (1 - encoder_tau) * target_param.data
            )
