import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel
from ..layers.channel_mlp import ChannelMLP
from ..layers.wno_block import WNOBlocks

class WNO(BaseModel):
    """
    WNO: Wavelet Neural Operator.
    """
    def __init__(
        self,
        n_modes,
        in_channels,
        out_channels,
        hidden_channels=64,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        wno_level=2,
        wno_wavelet='haar',
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        channel_mlp_dropout=0,
        non_linearity=F.gelu,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        **kwargs
    ):
        super().__init__()
        self.n_modes = n_modes
        self.n_dim = len(n_modes)
        self.hidden_channels = hidden_channels
        
        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            n_dim=self.n_dim
        )
        
        self.wno_blocks = WNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=n_layers,
            wno_level=wno_level,
            wno_wavelet=wno_wavelet,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_dropout=channel_mlp_dropout,
            non_linearity=non_linearity,
            norm=norm,
            ada_in_features=ada_in_features,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip
        )
        
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity
        )

    def forward(self, x, **kwargs):
        x = self.lifting(x)
        for i in range(self.wno_blocks.n_layers):
            x = self.wno_blocks(x, i)
        x = self.projection(x)
        return x
