import torch
import torch.nn.functional as F
import warnings
from .base_model import BaseModel
from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.wno_block import WNOBlocks
from ..layers.gno_block import GNOBlock
from ..layers.gno_weighting_functions import dispatch_weighting_fn

class GIWNO(BaseModel):
    """
    GIWNO: Geometry-informed Wavelet Neural Operator.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_feature_channels=None,
        projection_channel_ratio=4,
        gno_coord_dim=3,
        in_gno_radius=0.033,
        out_gno_radius=0.033,
        in_gno_transform_type="linear",
        out_gno_transform_type="linear",
        gno_weighting_function=None,
        gno_weight_function_scale=1,
        in_gno_pos_embed_type="transformer",
        out_gno_pos_embed_type="transformer",
        wno_in_channels=3,
        wno_n_modes=(16, 16),
        wno_hidden_channels=64,
        wno_lifting_channel_ratio=2,
        wno_n_layers=4,
        wno_level=2,
        wno_wavelet='haar',
        # Other GNO Params
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.gelu,
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        out_gno_tanh=None,
        # Other FNO/WNO Params
        wno_use_channel_mlp=True,
        wno_channel_mlp_dropout=0,
        wno_channel_mlp_expansion=0.5,
        wno_non_linearity=F.gelu,
        wno_norm=None,
        wno_ada_in_features=4,
        wno_ada_in_dim=1,
        wno_preactivation=False,
        wno_skip="linear",
        wno_channel_mlp_skip="soft-gating",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_feature_channels = latent_feature_channels
        self.gno_coord_dim = gno_coord_dim
        self.wno_hidden_channels = wno_hidden_channels

        self.lifting_channels = wno_lifting_channel_ratio * wno_hidden_channels

        if in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = wno_in_channels

        self.wno_in_channels = in_gno_out_channels
        if latent_feature_channels is not None:
            self.wno_in_channels += latent_feature_channels

        self.in_coord_dim = len(wno_n_modes)
        self.gno_out_coord_dim = len(wno_n_modes)

        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        self.in_coord_dim_reverse_order = [j + 2 for j in self.in_coord_dim_forward_order]

        self.in_gno = GNOBlock(
            in_channels=in_channels,
            out_channels=in_gno_out_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=in_gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            radius=in_gno_radius,
            reduction="mean",
            weighting_fn=None,
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=in_gno_transform_type,
            use_torch_scatter_reduce=gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        self.lifting = ChannelMLP(
            in_channels=self.wno_in_channels,
            hidden_channels=self.lifting_channels,
            out_channels=wno_hidden_channels,
            n_layers=2,
        )

        self.wno_blocks = WNOBlocks(
            n_modes=wno_n_modes,
            in_channels=wno_hidden_channels,
            out_channels=wno_hidden_channels,
            n_layers=wno_n_layers,
            wno_level=wno_level,
            wno_wavelet=wno_wavelet,
            use_channel_mlp=wno_use_channel_mlp,
            channel_mlp_expansion=wno_channel_mlp_expansion,
            channel_mlp_dropout=wno_channel_mlp_dropout,
            non_linearity=wno_non_linearity,
            norm=wno_norm,
            ada_in_features=wno_ada_in_features,
            preactivation=wno_preactivation,
            fno_skip=wno_skip,
            channel_mlp_skip=wno_channel_mlp_skip,
        )

        if gno_weighting_function is not None:
            weight_fn = dispatch_weighting_fn(
                gno_weighting_function,
                sq_radius=out_gno_radius**2,
                scale=gno_weight_function_scale,
            )
        else:
            weight_fn = None
            
        self.gno_out = GNOBlock(
            in_channels=wno_hidden_channels,
            out_channels=wno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=out_gno_radius,
            reduction="sum",
            weighting_fn=weight_fn,
            pos_embedding_type=out_gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=out_gno_transform_type,
            use_torch_scatter_reduce=gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        projection_channels = projection_channel_ratio * wno_hidden_channels
        self.projection = ChannelMLP(
            in_channels=wno_hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=wno_non_linearity,
        )

    def latent_embedding(self, in_p):
        # in_p : (batch, n_1, ..., n_k, c)
        in_p = in_p.permute(0, len(in_p.shape) - 1, *list(range(1, len(in_p.shape)-1)))
        in_p = self.lifting(in_p)
        for idx in range(self.wno_blocks.n_layers):
            in_p = self.wno_blocks(in_p, idx)
        return in_p

    def forward(
        self,
        input_geom,
        latent_queries,
        output_queries,
        x=None,
        latent_features=None,
        **kwargs,
    ):
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        input_geom = input_geom.squeeze(0)
        latent_queries = latent_queries.squeeze(0)

        in_p = self.in_gno(
            y=input_geom, x=latent_queries.view((-1, latent_queries.shape[-1])), f_y=x
        )

        grid_shape = latent_queries.shape[:-1]
        in_p = in_p.view((batch_size, *grid_shape, -1))

        if latent_features is not None:
            if latent_features.shape[0] == 1 and batch_size > 1:
                latent_features = latent_features.expand(batch_size, *latent_features.shape[1:])
            in_p = torch.cat((in_p, latent_features), dim=-1)
            
        latent_embed = self.latent_embedding(in_p=in_p)

        latent_embed = latent_embed.permute(0, *self.in_coord_dim_reverse_order, 1).reshape(batch_size, -1, self.wno_hidden_channels)

        if isinstance(output_queries, dict):
            out = {}
            for key, out_p in output_queries.items():
                out_p = out_p.squeeze(0)
                sub_output = self.gno_out(
                    y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                    x=out_p,
                    f_y=latent_embed,
                )
                sub_output = sub_output.permute(0, 2, 1)
                sub_output = self.projection(sub_output).permute(0, 2, 1)
                out[key] = sub_output
        else:
            output_queries = output_queries.squeeze(0)
            out = self.gno_out(
                y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                x=output_queries,
                f_y=latent_embed,
            )
            out = out.permute(0, 2, 1)
            out = self.projection(out).permute(0, 2, 1)

        return out
