import math
import torch
import torch.nn as nn

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from detectron2.layers import CNNBlockBase, Conv2d, get_norm


class SimpleFeaturePyramid(Backbone):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()
        # assert isinstance(net, Backbone)

        self.scale_factors = scale_factors

        # input_shapes = net.output_shape()
        ##strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        # _assert_strides_are_log2_contiguous(strides)

        dim = in_channels  # input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        dim // 2, dim // 4, kernel_size=2, stride=2
                    ),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
                ]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(
                    f"scale_factor={scale} is not supported yet."
                )

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = idx  # int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.top_block = top_block

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        features = x
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            top_block_in_feature = results[-1]
            results.extend(self.top_block(top_block_in_feature))

        return {f: res for f, res in enumerate(results)}
