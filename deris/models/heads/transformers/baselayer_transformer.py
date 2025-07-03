import copy
from torch import nn
from torch.nn import functional as F

class GlobalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.mapmlp = self.build_cpb_mlp(1, rpe_hidden_dim, num_heads)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim, bias=False)
        )
        return cpb_mlp

    def forward(
        self,
        query,
        k_input_flatten,
        v_input_flatten,
        input_padding_mask=None,
        attn_mask=None,  # B,1,N
    ):

        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn_mask = self.mapmlp(attn_mask.unsqueeze(-1)).permute(0, 3, 1, 2)  # (B,nhead,num_query,Nt)
            assert attn.shape == attn_mask.shape
            # attn =attn + attn_mask

        if input_padding_mask is not None:
            # attn += input_padding_mask[:, None, None] * -100
            input_mask = input_padding_mask.reshape(input_padding_mask.size(0), 1, 1, -1).repeat(
                1, self.num_heads, 1, 1
            )
            attn += input_mask * -100
        attn = self.softmax(attn)
        if attn_mask is not None:
            attn_mask = self.softmax(attn_mask)
            attn = (attn + attn_mask) / 2
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GlobalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        norm_type="post_norm",
    ):
        super().__init__()

        self.norm_type = norm_type

        # global cross attention
        self.cross_attn = GlobalCrossAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        # self attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[
            0
        ].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
            cross_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_post(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[
            0
        ].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
            cross_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        if self.norm_type == "pre_norm":
            return self.forward_pre(
                tgt, query_pos, src, src_pos_embed, src_padding_mask, self_attn_mask, cross_attn_mask
            )
        if self.norm_type == "post_norm":
            return self.forward_post(
                tgt, query_pos, src, src_pos_embed, src_padding_mask, self_attn_mask, cross_attn_mask
            )