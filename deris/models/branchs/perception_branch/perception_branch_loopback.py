from addict import Dict
import torch
from torch import nn
import pickle

from deris.models.builder import BRANCHS
from deris.models.branchs.perception_branch.mask_config.config import get_mask_config
from .Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
)
from deris.models.vis_encs.swin_trans import build_swin_b, build_swin_l, build_swin_s, build_swin_t


@BRANCHS.register_module()
class PerceptionBranchLoopBack(nn.Module):
    def __init__(
        self,
        mask_config,
        hidden_size,
        num_queries=100,
        dec_layers=9,
        pretrain_weights=None,
        freeze_hier_backbone=False,
        pixel_decoder_load_pretrain=True,
        prediction_load_pretrain=True,
        freeze_hier_pixel_decoder=False,
    ):
        super().__init__()
        self.mask_decoder_cfg = get_mask_config(config=mask_config)
        swin_type = getattr(self.mask_decoder_cfg, "swin_type", "base")
        if swin_type == "base":
            self.hier_vision_encoder = build_swin_b(pretrain_weights)
        elif swin_type == "tiny":
            self.hier_vision_encoder = build_swin_t(pretrain_weights)
        elif swin_type == "small":
            self.hier_vision_encoder = build_swin_s(pretrain_weights)
        elif swin_type == "large":
            self.hier_vision_encoder = build_swin_l(pretrain_weights)

        if freeze_hier_backbone:
            for param in self.hier_vision_encoder.parameters():
                param.requires_grad = False

        self.dec_layers = dec_layers
        self.pixel_decoder_load_pretrain = pixel_decoder_load_pretrain
        self.prediction_load_pretrain = prediction_load_pretrain

        self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = num_queries
        self.hier_vision_encoder.hidden_size = hidden_size
        self.freeze_hier_pixel_decoder = freeze_hier_pixel_decoder
        self.initial_mask_module(pretrained_path=pretrain_weights)
        self.nt_embed = nn.Linear(hidden_size, 1)
        self.perception_query_embed = nn.Embedding(num_queries, hidden_size)

    def get_vision_tower_feature(self, images):
        features = self.hier_vision_encoder(images)
        features_dict = {
            "res2": features[0],  # bs, 128, 256, 256
            "res3": features[1],  # bs, 256, 128, 128
            "res4": features[2],  # bs, 512, 64, 64
            "res5": features[3],  # bs, 1024, 32, 32
        }
        return features_dict

    def initial_mask_module(self, pretrained_path=None):

        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        # self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith(".pkl"):
                with open(pretrained_path, "rb") as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt["model"], "sem_seg_head.pixel_decoder")
            predictor_weights = get_w(ckpt["model"], "sem_seg_head.predictor")
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            # deal some diff keys
            change_w(pixel_decoder_weights, "adapter_1.weight", "adapter_1.0.weight")
            change_w(pixel_decoder_weights, "adapter_1.norm.weight", "adapter_1.1.weight")
            change_w(pixel_decoder_weights, "adapter_1.norm.bias", "adapter_1.1.bias")
            change_w(pixel_decoder_weights, "layer_1.weight", "layer_1.0.weight")
            change_w(pixel_decoder_weights, "layer_1.norm.weight", "layer_1.1.weight")
            change_w(pixel_decoder_weights, "layer_1.norm.bias", "layer_1.1.bias")
            if "static_query.weight" in predictor_weights:
                change_w(predictor_weights, "static_query.weight", "query_feat.weight")
            if (
                predictor_weights["query_embed.weight"].shape[0]
                != self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ):
                predictor_weights["query_embed.weight"] = predictor_weights["query_embed.weight"][
                    : self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, :
                ]
            if self.pixel_decoder_load_pretrain:
                diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights, strict=False)
                print(diff_pixel_msg)
            if self.prediction_load_pretrain:
                diff_predictor_msg = self.predictor.load_state_dict(predictor_weights, strict=False)
                print(diff_predictor_msg)
        if self.freeze_hier_pixel_decoder:
            for param in self.pixel_decoder.parameters():
                param.requires_grad = False

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = (
            cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        )  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout,
            transformer_nheads,
            transformer_dim_feedforward,
            transformer_enc_layers,
            conv_dim,
            mask_dim,
            transformer_in_features,
            common_stride,
        )
        return pixel_decoder

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = self.dec_layers if self.dec_layers is not None else cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(
            in_channels,
            hidden_dim,
            num_queries,
            nheads,
            dim_feedforward,
            dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_project,
            seg_norm,
            seg_proj,
            seg_fuse_score,
        )
        return predictor

    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [
            int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2**i)
            for i in range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))
        ]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {"channel": out_feature_channels[name], "stride": out_feature_strides[name]}
            )
        return backbone_feature_shape

    def forward(self, img, img_feat, txt_feat, text_padding_mask, perception_queries=None):
        if perception_queries is None:
            perception_queries = self.perception_query_embed.weight.unsqueeze(0).repeat(img.shape[0], 1, 1)

        image_features = self.get_vision_tower_feature(img)

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features
        )

        mask_outputs = self.predictor(
            multi_scale_features, mask_features, None, perception_queries, [txt_feat, text_padding_mask, img_feat]
        )

        return mask_outputs
