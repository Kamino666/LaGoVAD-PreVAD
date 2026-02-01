import torch
import torch.nn as nn
from collections import OrderedDict

from .modeling_vadclip import QuickGELU


class CoAttnFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.text_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored
        attn_t_feat = self.text_norm1(t_feat + self.text_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        attn_v_feat = self.vis_norm1(v_feat + self.vis_attn(v_feat, t_feat, t_feat)[0])

        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))
        attn_v_feat = self.vis_norm2(attn_v_feat + self.vis_ffn(attn_v_feat))

        return attn_v_feat, attn_t_feat


class UniAttnFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.text_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored
        attn_t_feat = self.text_norm1(t_feat + self.text_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))

        return v_feat, attn_t_feat


class YWFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.text_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.empty(1).fill_(3.0))  # re-scale dot-product similarity

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored

        # v2t fusion
        attn_t_feat = self.text_norm1(t_feat + self.v2t_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        # t2v re-weight
        sim_mat = torch.einsum('bce,bte->btc', self.text_k_proj(attn_t_feat), self.vis_q_proj(v_feat))  # [B,T,C]
        mask_sim_mat = torch.where(attn_mask == 1, -100, 0)
        sim_mat = sim_mat * mask_sim_mat.unsqueeze(-1)
        sim_weight = torch.sigmoid(sim_mat.max(dim=-1)[0] * self.scale)  # [B,T]
        attn_v_feat = self.vis_norm1(self.vis_v_proj(v_feat) * sim_weight.unsqueeze(-1))  # [B,T,E]
        # ffn
        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))
        attn_v_feat = self.vis_norm2(attn_v_feat + self.vis_ffn(attn_v_feat))

        return attn_v_feat, attn_t_feat


class AsymFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.text_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.empty(1).fill_(3.0))  # re-scale dot-product similarity

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored

        # v2t fusion
        attn_t_feat = self.text_norm1(t_feat + self.v2t_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        # t2v re-weight
        sim_mat = torch.einsum('bce,bte->btc', self.text_k_proj(attn_t_feat), self.vis_q_proj(v_feat))  # [B,T,C]
        mask_sim_mat = torch.where(attn_mask == 1, -100, 0)
        sim_mat = sim_mat * mask_sim_mat.unsqueeze(-1)
        # -> max pooling [B,C] -> argmax [B]
        match_idx = sim_mat.max(dim=1)[0].argmax(dim=-1)
        # -> select matched sim [B,T]
        sim_mat_match = sim_mat[:, :, match_idx]
        sim_weight = torch.sigmoid(sim_mat_match * self.scale)  # [B,T]
        attn_v_feat = self.vis_norm1(self.vis_v_proj(v_feat) * sim_weight.unsqueeze(-1))  # [B,T,E]
        # ffn
        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))
        attn_v_feat = self.vis_norm2(attn_v_feat + self.vis_ffn(attn_v_feat))

        return attn_v_feat, attn_t_feat


class FusionV1(nn.Module):
    def __init__(self, fusion_type: str = 'co_attn', d_model=512, nhead=8, num_layers=2):
        super().__init__()
        if fusion_type == 'co_attn':
            self.layers = nn.ModuleList([CoAttnFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        elif fusion_type == 'yw':
            self.layers = nn.ModuleList([YWFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        elif fusion_type == 'asym':
            self.layers = nn.ModuleList([AsymFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        elif fusion_type == 'uni':
            self.layers = nn.ModuleList([UniAttnFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        else:
            raise NotImplementedError

    def forward(self, v_feat, t_feat, lengths):
        # v_feat: [B, T, E]
        # t_feat: [C, E]
        # lengths: [B]
        bs = v_feat.shape[0]
        attn_mask = torch.arange(v_feat.shape[1], device=v_feat.device)[None, :] >= lengths[:, None]

        for layer in self.layers:
            v_feat, t_feat = layer(v_feat, t_feat, attn_mask)

        return v_feat, t_feat


class VadCLIPFusion(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

    def forward(self, v_feat, t_feat, lengths, bin_logits):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # lengths: [B]
        # bin_logits: [B, T]
        agg_v_feat = torch.einsum('bte,bt->be', v_feat, bin_logits)

        visual_attn = (agg_v_feat / agg_v_feat.norm(dim=-1, keepdim=True))[:, None, :]  # [B,1,E]
        visual_attn = visual_attn.expand(visual_attn.shape[0], t_feat.shape[1], visual_attn.shape[2])  # [B,C,E]
        text_features = t_feat + visual_attn
        text_features = text_features + self.mlp1(text_features)

        return text_features
