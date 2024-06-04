from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint


def checkpoint_seq(
    functions,
    x,
    every=1,
    flatten=False,
    skip_last=False,
    preserve_rng_state=True,
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """

    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(
            run_function(start, end, functions),
            x,
            preserve_rng_state=preserve_rng_state,
        )
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        num_frms: int = 100,
        temporal_downrate: int = 1,
        num_queries: int = 32,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        pre_norm: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_queries = num_queries
        act_layer = act_layer or nn.GELU
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        embed_len = num_frms
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_len, embed_dim) * 0.02
        )
        trunc_normal_(self.pos_embed, std=0.02)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        if temporal_downrate > 1:
            self.temporal_proj = nn.Linear(
                embed_dim * temporal_downrate, embed_dim, bias=False
            )

        else:
            self.temporal_proj = nn.Linear(
                embed_dim * num_queries, embed_dim, bias=False
            )
            self.num_queries = 1

        self.temporal_downrate = temporal_downrate

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def _pos_embed(self, x):
        if x.ndim == 3:
            # B, T, C
            T = x.shape[1]
            if T == self.pos_embed.shape[1]:
                x = x + self.pos_embed
            else:
                # special case: SOT and concat query
                num_query_in_tgt = self.num_queries
                num_query_in_ref = T - num_query_in_tgt
                x_ref = x[:, :num_query_in_ref] + self.pos_embed[:, :1]
                x_tgt = x[:, num_query_in_ref:] + self.pos_embed[:, 1:]
                x = torch.cat([x_ref, x_tgt], dim=1)

        else:
            # B, T, 1 -> B, T, 32, C
            pos_embed = self.pos_embed.unsqueeze(2).repeat(
                1, 1, self.num_queries, 1
            )
            x = x + pos_embed[:, : x.shape[1]]
            x = x.view(x.shape[0], -1, x.shape[3])

        return x

    def forward(self, x):
        B, T, num_q, _ = x.shape
        if self.temporal_downrate > 1:
            assert num_q == self.num_queries, "input feature has wrong size"
            # B, T, 32, C -> B, 32, T/down, C*down
            x = (
                x.permute(0, 2, 1, 3)
                .contiguous()
                .view(B, num_q, T // self.temporal_downrate, -1)
            )
            # B, 32, T/down, C*down -> B, 32, T/down, C
            x = self.temporal_proj(x)
            # B, 32, T/down, C -> B, T/down, 32, C
            x = x.permute(0, 2, 1, 3).contiguous()

        else:
            # B, T, 32, C -> B, T, 32xC
            x = x.view(B, T, -1)
            # B, T, 32xC -> B, T, C
            x = self.temporal_proj(x)

        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        return x
