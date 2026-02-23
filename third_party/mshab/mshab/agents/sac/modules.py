import torch
import torch.nn as nn

from mshab.agents.sac.misc import gaussian_logprob, get_out_shape, squash, weight_init


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.out_dim = out_dim
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class SharedCNN(nn.Module):
    def __init__(self, pixel_obs_shape, features, filters, strides, padding):
        super().__init__()
        assert len(pixel_obs_shape) == 3

        in_features = [pixel_obs_shape[0]] + features
        out_features = features
        layers = []
        for i, (in_f, out_f, filter_, stride) in enumerate(
            zip(in_features, out_features, filters, strides)
        ):
            layers.append(nn.Conv2d(in_f, out_f, filter_, stride, padding=padding))
            if i < len(filters) - 1:
                layers.append(nn.ReLU())
        layers.append(Flatten())
        self.layers = nn.Sequential(*layers)

        self.out_dim = get_out_shape(pixel_obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, pixels: torch.Tensor):
        # NOTE (arth): a bit unclean? basically with we get concat'd pixel images,
        # it's easier to squash them here (once) than when actor/critic called in Algo
        # (multiple times). generally better to keep the modules separate from
        # this sort of logic, but this is easier
        with torch.no_grad():
            if len(pixels.shape) == 5:
                b, fs, d, h, w = pixels.shape
                pixels = pixels.view(b, fs * d, h, w).contiguous()
        return self.layers(pixels)


class Encoder(nn.Module):
    """Convolutional encoder of pixels and state observations."""

    def __init__(
        self,
        cnns: nn.ModuleDict,
        pixels_projections: nn.ModuleDict,
        state_projection: RLProjection,
    ):
        super().__init__()
        self.cnns = cnns
        self.pixels_projections = pixels_projections
        self.state_projection = state_projection
        self.out_dim = (
            sum([pix_proj.out_dim for pix_proj in self.pixels_projections.values()])
            + state_projection.out_dim
        )

    def forward(self, pixels, state, detach=False):
        pencs = [(k, cnn(pixels[k])) for k, cnn in self.cnns.items()]
        if detach:
            pencs = [(k, p.detach()) for k, p in pencs]
        pixels = torch.cat([self.pixels_projections[k](p) for k, p in pencs], dim=1)
        state = self.state_projection(state)
        return torch.cat([pixels, state], dim=1)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(self, encoder, action_dim, hidden_dims, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder

        in_dims = [self.encoder.out_dim] + hidden_dims
        out_dims = hidden_dims + [2 * action_dim]
        mlp_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(weight_init)

    def forward(
        self, pixels, state, compute_pi=True, compute_log_pi=True, detach=False
    ):
        x = self.encoder(pixels, state, detach=detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std


class Critic(nn.Module):
    def __init__(
        self, encoder, action_dim, hidden_dims, layer_norm=False, dropout=None
    ):
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(
            self.encoder.out_dim,
            action_dim,
            hidden_dims,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.Q2 = QFunction(
            self.encoder.out_dim,
            action_dim,
            hidden_dims,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.apply(weight_init)

    def forward(self, pixels, state, action, detach=False):
        x = self.encoder(pixels, state, detach=detach)
        return self.Q1(x, action), self.Q2(x, action)


class QFunction(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dims, layer_norm=False, dropout=None
    ):
        super().__init__()

        in_dims = [obs_dim + action_dim] + hidden_dims
        out_dims = hidden_dims + [1]
        mlp_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                if dropout is not None and dropout > 0:
                    mlp_layers.append(nn.Dropout(p=dropout))
                if layer_norm:
                    mlp_layers.append(nn.LayerNorm(out_dim))
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.mlp(obs_action)
