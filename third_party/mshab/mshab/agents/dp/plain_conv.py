from typing import Tuple

import torch
import torch.nn as nn


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        out_dim=256,
        pool_feature_map=False,
        last_act=True,  # True for ConvBody, False for CNN
    ):
        super().__init__()

        C, H, W = img_shape

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            _test_out = self.pool(self.cnn(torch.randn(1, C, H, W))).flatten(1)
            output_dim = _test_out.size(1)
        else:
            self.pool = None
            _test_out = self.cnn(torch.randn(1, C, H, W)).flatten(1)
            output_dim = _test_out.size(1)

        self.fc = make_mlp(output_dim, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        enc = self.cnn(image)
        if self.pool is not None:
            enc = self.pool(enc)
        return self.fc(enc.flatten(1))
