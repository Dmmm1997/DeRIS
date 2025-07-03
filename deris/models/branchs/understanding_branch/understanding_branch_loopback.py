from torch import nn
from deris.models.builder import BRANCHS, build_vis_enc
from deris.models.heads.transformers.transformer import DetrTransformerDecoder


class MultiGrainAligner(nn.Module):
    def __init__(self, hidden_channels=256, layer=3):
        super(MultiGrainAligner, self).__init__()
        self.TextCorrelation = DetrTransformerDecoder(
            embed_dim=hidden_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=512,
            ffn_dropout=0.1,
            num_layers=layer,
            return_intermediate=False,
            post_norm=True,
            batch_first=True,
        )

    def forward(self, query_feat, pred_dict, img_feat, text_feat, text_valid_mask):
        roi_features = query_feat  # [B, N, C]
        out_feat, _ = self.TextCorrelation(
            query=roi_features,
            key=text_feat,
            value=text_feat,
            key_pos=None,
            key_padding_mask=text_valid_mask.bool(),
        )
        return out_feat[-1], None


@BRANCHS.register_module()
class UnderstandingBranchLoopBack(nn.Module):
    def __init__(
        self,
        vis_enc,
        vis_enc_outdim,
        hidden_channels=256,
        freeze_vis_enc=False,
        num_classes=2,
        layer=3,
        num_queries=20,
    ):
        super().__init__()
        self.vis_enc = build_vis_enc(vis_enc)
        self.vis_embed = nn.Linear(vis_enc_outdim, hidden_channels)
        self.text_embed = nn.Linear(vis_enc_outdim, hidden_channels)
        self.roialigner = MultiGrainAligner(hidden_channels=hidden_channels, layer=layer)
        self.perception_embed = nn.Linear(vis_enc_outdim, hidden_channels)
        self.logit_embed = nn.Linear(hidden_channels, num_classes)
        self.nt_embed = nn.Linear(hidden_channels, 1)
        if freeze_vis_enc:
            for param in self.vis_enc.parameters():
                param.requires_grad = False
        self.init_query = nn.Embedding(num_queries, vis_enc_outdim)

    def pre_forward(self, img, ref_expr_inds, text_attention_mask):
        B, _, H, W = img.shape
        query_feat = self.init_query.weight.unsqueeze(0).repeat(img.shape[0], 1, 1)
        img_feat, text_feat, cls_feat = self.vis_enc(img, ref_expr_inds, text_attention_mask)
        query_feat = self.perception_embed(query_feat)
        img_feat = self.vis_embed(img_feat)
        text_feat = self.text_embed(text_feat)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // 16, W // 16)
        feat_dict = {"img_feat": img_feat, "text_feat": text_feat, "query_feat": query_feat, "cls_feat": cls_feat}
        return feat_dict

    def post_forward(self, perception_queries, pred_dict, img_feat, text_feat, text_attention_mask, cls_feat):
        pred_global_mask = None
        query_feat, pred_logits = self.roialigner(
            perception_queries, pred_dict, img_feat, text_feat, text_attention_mask
        )
        pred_logits = self.logit_embed(query_feat)  # [B,N,2]
        pred_existence = self.nt_embed(query_feat).mean(-2)  # BNC-> B*N*1 -> B*1
        return {
            "pred_logits": pred_logits,
            "query_feat": query_feat,
            "pred_existence": pred_existence,
            "pred_global_mask": pred_global_mask,
        }

    def extract_visual_language(self, img, ref_expr_inds, text_attention_mask=None):
        x, y, c = self.vis_enc(img, ref_expr_inds, text_attention_mask)
        return x, y, c
