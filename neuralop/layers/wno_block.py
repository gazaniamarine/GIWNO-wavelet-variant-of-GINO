from typing import List, Union
import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .normalization_layers import AdaIN, InstanceNorm, BatchNorm
from .skip_connections import skip_connection
from .wavelet_convolution import WaveletConv

class WNOBlocks(nn.Module):
    """WNOBlocks implements a sequence of Wavelet layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        n_layers=1,
        wno_level=2,
        wno_wavelet='haar',
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip

        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features
        self.non_linearity = non_linearity

        # One conv per layer
        self.convs = nn.ModuleList(
            [
                WaveletConv(
                    self.in_channels,
                    self.out_channels,
                    self._n_modes,
                    level=wno_level,
                    wavelet=wno_wavelet
                )
                for i in range(n_layers)
            ]
        )

        if fno_skip is not None:
            self.fno_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=fno_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.fno_skips = None

        if self.use_channel_mlp:
            self.channel_mlp = nn.ModuleList(
                [
                    ChannelMLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * channel_mlp_expansion),
                        dropout=channel_mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            if channel_mlp_skip is not None:
                self.channel_mlp_skips = nn.ModuleList(
                    [
                        skip_connection(
                            self.in_channels,
                            self.out_channels,
                            skip_type=channel_mlp_skip,
                            n_dim=self.n_dim,
                        )
                        for _ in range(n_layers)
                    ]
                )
            else:
                self.channel_mlp_skips = None

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [InstanceNorm() for _ in range(n_layers * self.n_norms)]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "batch_norm":
            self.norm = nn.ModuleList(
                [
                    BatchNorm(n_dim=self.n_dim, num_features=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(f"Got norm={norm}")

    def set_ada_in_embeddings(self, *embeddings):
        if self.norm is not None:
            if len(embeddings) == 1:
                for norm in self.norm:
                    norm.set_embedding(embeddings[0])
            else:
                for norm, embedding in zip(self.norm, embeddings):
                    norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes
