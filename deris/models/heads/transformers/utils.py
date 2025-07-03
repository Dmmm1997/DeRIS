# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py
# ------------------------------------------------------------------------------------------------
import math
import copy
import warnings
from typing import List
import torch
import torch.nn as nn
from typing import Optional
from torch.nn.init import constant_, xavier_uniform_
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class BaseTransformerLayer(nn.Module):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.

    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
        self,
        attn: List[nn.Module],
        ffn: nn.Module,
        norm: nn.Module,
        operation_order: tuple = None,
    ):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: List[torch.Tensor] = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query, decoder_attn_map = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                if isinstance(query, tuple):
                    query = self.norms[norm_index](query[0])
                else:
                    query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_map = None
                if isinstance(query, tuple):
                    attn_map = query[1]
                    identity = query[0]
                    query = query[0]
                else:
                    identity = query
                attn_index += 1

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query, attn_map


class TransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder, which will copy
    the passed `transformer_layers` module `num_layers` time or save the passed
    list of `transformer_layers` as parameters named ``self.layers``
    which is the type of ``nn.ModuleList``.
    The users should inherit `TransformerLayerSequence` and implemente their
    own forward function.

    Args:
        transformer_layers (list[BaseTransformerLayer] | BaseTransformerLayer): A list
            of BaseTransformerLayer. If it is obj:`BaseTransformerLayer`, it
            would be repeated `num_layers` times to a list[BaseTransformerLayer]
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(
        self,
        transformer_layers=None,
        num_layers=None,
    ):
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformer_layers, nn.Module):
            for _ in range(num_layers):
                self.layers.append(copy.deepcopy(transformer_layers))
        else:
            assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is" f"missing in {self.__class__.__name__}.")
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out, decoder_attn_map = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        return identity + self.proj_drop(out), decoder_attn_map


class FFN(nn.Module):
    """The implementation of feed-forward networks (FFNs)
    with identity connection.

    Args:
        embed_dim (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_dim (int): The hidden dimension of FFNs.
            Defaults: 1024.
        output_dim (int): The output feature dimension of FFNs.
            Default: None. If None, the `embed_dim` will be used.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        activation (nn.Module): The activation layer used in FFNs.
            Default: nn.ReLU(inplace=True).
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
    """

    def __init__(
        self,
        embed_dim=256,
        feedforward_dim=1024,
        output_dim=None,
        num_fcs=2,
        activation=nn.ReLU(inplace=True),
        ffn_drop=0.0,
        fc_bias=True,
        add_identity=True,
    ):
        super(FFN, self).__init__()
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = []
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_dim, bias=fc_bias),
                    self.activation,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_dim
        layers.append(nn.Linear(feedforward_dim, output_dim, bias=fc_bias))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None) -> torch.Tensor:
        """Forward function of `FFN`.

        Args:
            x (torch.Tensor): the input tensor used in `FFN` layers.
            identity (torch.Tensor): the tensor with the same shape as `x`,
                which will be used for identity addition. Default: None.
                if None, `x` will be used.

        Returns:
            torch.Tensor: the forward results of `FFN` layer
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


# class MultiScaleDeformableAttention(nn.Module):
#     """Multi-Scale Deformable Attention Module used in Deformable-DETR

#     `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
#     <https://arxiv.org/pdf/2010.04159.pdf>`_.

#     Args:
#         embed_dim (int): The embedding dimension of Attention. Default: 256.
#         num_heads (int): The number of attention heads. Default: 8.
#         num_levels (int): The number of feature map used in Attention. Default: 4.
#         num_points (int): The number of sampling points for each query
#             in each head. Default: 4.
#         img2col_steps (int): The step used in image_to_column. Defualt: 64.
#             dropout (float): Dropout layer used in output. Default: 0.1.
#         batch_first (bool): if ``True``, then the input and output tensor will be
#             provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
#     """

#     def __init__(
#         self,
#         embed_dim: int = 256,
#         num_heads: int = 8,
#         num_levels: int = 4,
#         num_points: int = 4,
#         img2col_step: int = 64,
#         dropout: float = 0.1,
#         batch_first: bool = False,
#     ):
#         super().__init__()
#         if embed_dim % num_heads != 0:
#             raise ValueError(
#                 "embed_dim must be divisible by num_heads, but got {} and {}".format(
#                     embed_dim, num_heads
#                 )
#             )
#         head_dim = embed_dim // num_heads

#         self.dropout = nn.Dropout(dropout)
#         self.batch_first = batch_first

#         if not _is_power_of_2(head_dim):
#             warnings.warn(
#                 """
#                 You'd better set d_model in MSDeformAttn to make sure that
#                 each dim of the attention head a power of 2, which is more efficient.
#                 """
#             )

#         self.im2col_step = img2col_step
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_levels = num_levels
#         self.num_points = num_points
#         # n_heads * n_points and n_levels for multi-level feature inputs
#         self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
#         self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
#         self.value_proj = nn.Linear(embed_dim, embed_dim)
#         self.output_proj = nn.Linear(embed_dim, embed_dim)

#         self.init_weights()

#     def init_weights(self):
#         """
#         Default initialization for Parameters of Module.
#         """
#         constant_(self.sampling_offsets.weight.data, 0.0)
#         thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
#             2.0 * math.pi / self.num_heads
#         )
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (
#             (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
#             .view(self.num_heads, 1, 1, 2)
#             .repeat(1, self.num_levels, self.num_points, 1)
#         )
#         for i in range(self.num_points):
#             grid_init[:, :, i, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.0)
#         constant_(self.attention_weights.bias.data, 0.0)
#         xavier_uniform_(self.value_proj.weight.data)
#         constant_(self.value_proj.bias.data, 0.0)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.0)

#     def forward(
#         self,
#         query: torch.Tensor,
#         key: Optional[torch.Tensor] = None,
#         value: Optional[torch.Tensor] = None,
#         identity: Optional[torch.Tensor] = None,
#         query_pos: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         reference_points: Optional[torch.Tensor] = None,
#         spatial_shapes: Optional[torch.Tensor] = None,
#         level_start_index: Optional[torch.Tensor] = None,
#         **kwargs
#     ) -> torch.Tensor:

#         """Forward Function of MultiScaleDeformableAttention

#         Args:
#             query (torch.Tensor): Query embeddings with shape
#                 `(num_query, bs, embed_dim)`
#             key (torch.Tensor): Key embeddings with shape
#                 `(num_key, bs, embed_dim)`
#             value (torch.Tensor): Value embeddings with shape
#                 `(num_key, bs, embed_dim)`
#             identity (torch.Tensor): The tensor used for addition, with the
#                 same shape as `query`. Default: None. If None, `query` will be
#                 used.
#             query_pos (torch.Tensor): The position embedding for `query`. Default: None.
#             key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
#                 indicating which elements within `key` to be ignored in attention.
#             reference_points (torch.Tensor): The normalized reference points
#                 with shape `(bs, num_query, num_levels, 2)`,
#                 all elements is range in [0, 1], top-left (0, 0),
#                 bottom-right (1, 1), including padding are.
#                 or `(N, Length_{query}, num_levels, 4)`, add additional
#                 two dimensions `(h, w)` to form reference boxes.
#             spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
#                 With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
#             level_start_index (torch.Tensor): The start index of each level. A tensor with
#                 shape `(num_levels, )` which can be represented as
#                 `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

#         Returns:
#             torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
#         """

#         if value is None:
#             value = query

#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             query = query + query_pos

#         if not self.batch_first:
#             # change to (bs, num_query ,embed_dims)
#             query = query.permute(1, 0, 2)
#             value = value.permute(1, 0, 2)

#         bs, num_query, _ = query.shape
#         bs, num_value, _ = value.shape

#         assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

#         # value projection
#         value = self.value_proj(value)
#         # fill "0" for the padding part
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], float(0))
#         # [bs, all hw, 256] -> [bs, all hw, 8, 32]
#         value = value.view(bs, num_value, self.num_heads, -1)
#         # [bs, all hw, 8, 4, 4, 2]: 8 heads, 4 level features, 4 sampling points, 2 offsets
#         sampling_offsets = self.sampling_offsets(query).view(
#             bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
#         )
#         # [bs, all hw, 8, 16]: 4 level 4 sampling points: 16 features total
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query, self.num_heads, self.num_levels * self.num_points
#         )
#         attention_weights = attention_weights.softmax(-1)
#         attention_weights = attention_weights.view(
#             bs,
#             num_query,
#             self.num_heads,
#             self.num_levels,
#             self.num_points,
#         )

#         # bs, num_query, num_heads, num_levels, num_points, 2
#         if reference_points.shape[-1] == 2:

#             # reference_points   [bs, all hw, 4, 2] -> [bs, all hw, 1, 4, 1, 2]
#             # sampling_offsets   [bs, all hw, 8, 4, 4, 2]
#             # offset_normalizer  [4, 2] -> [1, 1, 1, 4, 1, 2]
#             # references_points + sampling_offsets

#             offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#             sampling_locations = (
#                 reference_points[:, :, None, :, None, :]
#                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#             )
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = (
#                 reference_points[:, :, None, :, None, :2]
#                 + sampling_offsets
#                 / self.num_points
#                 * reference_points[:, :, None, :, None, 2:]
#                 * 0.5
#             )
#         else:
#             raise ValueError(
#                 "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
#                     reference_points.shape[-1]
#                 )
#             )

#         # the original impl for fp32 training
#         if torch.cuda.is_available() and value.is_cuda:
#             output = MultiScaleDeformableAttnFunction.apply(
#                 value.to(torch.float32) if value.dtype==torch.float16 else value,
#                 spatial_shapes,
#                 level_start_index,
#                 sampling_locations,
#                 attention_weights,
#                 self.im2col_step,
#             )
#         else:
#             output = multi_scale_deformable_attn_pytorch(
#                 value, spatial_shapes, sampling_locations, attention_weights
#             )

#         if value.dtype==torch.float16:
#             output=output.to(torch.float16)

#         output = self.output_proj(output)

#         if not self.batch_first:
#             output = output.permute(1, 0, 2)

#         return self.dropout(output) + identity, attention_weights


# def multi_scale_deformable_attn_pytorch(
#     value: torch.Tensor,
#     value_spatial_shapes: torch.Tensor,
#     sampling_locations: torch.Tensor,
#     attention_weights: torch.Tensor,
# ) -> torch.Tensor:

#     bs, _, num_heads, embed_dims = value.shape
#     _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
#     value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
#     sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#     for level, (H_, W_) in enumerate(value_spatial_shapes):
#         # bs, H_*W_, num_heads, embed_dims ->
#         # bs, H_*W_, num_heads*embed_dims ->
#         # bs, num_heads*embed_dims, H_*W_ ->
#         # bs*num_heads, embed_dims, H_, W_
#         value_l_ = (
#             value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
#         )
#         # bs, num_queries, num_heads, num_points, 2 ->
#         # bs, num_heads, num_queries, num_points, 2 ->
#         # bs*num_heads, num_queries, num_points, 2
#         sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
#         # bs*num_heads, embed_dims, num_queries, num_points
#         sampling_value_l_ = F.grid_sample(
#             value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
#         )
#         sampling_value_list.append(sampling_value_l_)
#     # (bs, num_queries, num_heads, num_levels, num_points) ->
#     # (bs, num_heads, num_queries, num_levels, num_points) ->
#     # (bs, num_heads, 1, num_queries, num_levels*num_points)
#     attention_weights = attention_weights.transpose(1, 2).reshape(
#         bs * num_heads, 1, num_queries, num_levels * num_points
#     )
#     output = (
#         (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
#         .sum(-1)
#         .view(bs, num_heads * embed_dims, num_queries)
#     )
#     return output.transpose(1, 2).contiguous()


# class MultiScaleDeformableAttnFunction(Function):
#     @staticmethod
#     def forward(
#         ctx,
#         value,
#         value_spatial_shapes,
#         value_level_start_index,
#         sampling_locations,
#         attention_weights,
#         im2col_step,
#     ):
#         ctx.im2col_step = im2col_step
#         output = _C.ms_deform_attn_forward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             ctx.im2col_step,
#         )
#         ctx.save_for_backward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#         )
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         (
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#         ) = ctx.saved_tensors
#         grad_value, grad_sampling_loc, grad_attn_weight = _C.ms_deform_attn_backward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             grad_output,
#             ctx.im2col_step,
#         )

#         return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


# def create_dummy_func(func, dependency, message=""):
#     """
#     When a dependency of a function is not available, create a dummy function which throws
#     ImportError when used.

#     Args:
#         func (str): name of the function.
#         dependency (str or list[str]): name(s) of the dependency.
#         message: extra message to print
#     Returns:
#         function: a function object
#     """
#     err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, func)
#     if message:
#         err = err + " " + message

#     if isinstance(dependency, (list, tuple)):
#         dependency = ",".join(dependency)

#     def _dummy(*args, **kwargs):
#         raise ImportError(err)

#     return _dummy

# try:
#     from detrex import _C
# except ImportError:
#     # TODO: register ops natively so there is no need to import _C.
#     _msg = "detrex is not compiled successfully, please build following the instructions!"
#     _args = ("detrex._C", _msg)
#     MultiScaleDeformableAttention = create_dummy_class(  # noqa
#         "MultiScaleDeformableAttention", *_args
#     )
