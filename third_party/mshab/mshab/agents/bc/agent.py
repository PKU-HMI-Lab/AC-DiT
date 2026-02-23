# CNN from https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

from typing import Dict

from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, sample_obs, single_act_shape):
        super().__init__()

        extractors = dict()

        extractor_out_features = 0
        feature_size = 1024

        pixel_obs: Dict[str, torch.Tensor] = sample_obs["pixels"]
        state_obs: torch.Tensor = sample_obs["state"]

        for k, pobs in pixel_obs.items():
            if len(pobs.shape) == 5:
                b, fs, d, h, w = pobs.shape
                pobs = pobs.reshape(b, fs * d, h, w)
            pobs_stack = pobs.size(1)
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=pobs_stack,
                    out_channels=24,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=24,
                    out_channels=36,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=36,
                    out_channels=48,
                    kernel_size=5,
                    stride=2,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=48,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flatten = cnn(pobs.float().cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors[k] = nn.Sequential(cnn, fc)
            extractor_out_features += feature_size

        # for state data we simply pass it through a single linear layer
        extractors["state"] = nn.Linear(state_obs.size(-1), feature_size)
        extractor_out_features += feature_size

        self.extractors = nn.ModuleDict(extractors)

        self.mlp = nn.Sequential(
            layer_init(nn.Linear(extractor_out_features, 2048)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, np.prod(single_act_shape)),
                std=0.01 * np.sqrt(2),
            ),
        )

    def forward(self, observations) -> torch.Tensor:
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if key == "state":
                encoded_tensor_list.append(extractor(state))
            else:
                with torch.no_grad():
                    pobs = pixels[key].float()
                    # pobs = 1 / (1 + pixels[key].float() / 400)
                    pobs = 1 - torch.tanh(pixels[key].float() / 1000)
                    if len(pobs.shape) == 5:
                        b, fs, d, h, w = pobs.shape
                        pobs = pobs.reshape(b, fs * d, h, w)
                encoded_tensor_list.append(extractor(pobs))
        return self.mlp(torch.cat(encoded_tensor_list, dim=1))
