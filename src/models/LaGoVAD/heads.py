import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Literal


class ConvScoreHead(nn.Module):
    def __init__(self, d_model: int,
                 num_layers: int = 3,
                 kernel_size: int = 3, activation: str = 'gelu', norm: str = 'layernorm'):
        super().__init__()
        n_pad = (kernel_size - 1) // 2
        dim_list = [d_model // (2 ** i) for i in range(num_layers)]  # 512 256 128
        self.convs = nn.ModuleList([
            nn.Conv1d(
                dim_list[i],
                dim_list[i + 1],
                kernel_size=kernel_size,
                padding=n_pad,
                padding_mode='replicate'
            )
            for i in range(num_layers - 1)
        ])
        self.convs.append(nn.Conv1d(dim_list[-1], 1, kernel_size=kernel_size,
                                    padding=n_pad, padding_mode='replicate'))

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise NotImplementedError
        if norm == 'layernorm':
            self.norm = nn.ModuleList([nn.LayerNorm(d) for d in dim_list[1:]])
        elif norm == 'batchnorm':
            self.norm = nn.ModuleList([nn.BatchNorm1d(d) for d in dim_list[1:]])
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # x: b c t
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:  # skip last activation
                x = self.norm[i](self.activation(x).permute(0, 2, 1)).permute(0, 2, 1)
        # x: b 1 t
        return x


class SimScoreHead(nn.Module):
    def __init__(self, temperature_init=0.2):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([1]).fill_(temperature_init))  # 1

    def forward(self, x_feat: Tensor, y_feat: Tensor):
        # x: [B,T,E]     y: [B,C,E]
        x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
        y_feat = y_feat / y_feat.norm(dim=-1, keepdim=True)
        sim = torch.einsum('bte,bce->btc', x_feat, y_feat)
        sim = sim / self.temperature  # B,T,C
        return sim


class ProjSimScoreHead(nn.Module):
    def __init__(self, dim_model, dim_proj, temperature_init=0.2):
        super().__init__()
        self.x_proj = nn.Linear(dim_model, dim_proj, bias=False)
        self.y_proj = nn.Linear(dim_model, dim_proj, bias=False)
        self.temperature = nn.Parameter(torch.ones([1]).fill_(temperature_init))  # 1

    def forward(self, x_feat: Tensor, y_feat: Tensor):
        # x: [B,T,E]     y: [B,C,E]
        x_feat = self.x_proj(x_feat)
        y_feat = self.y_proj(y_feat)
        x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
        y_feat = y_feat / y_feat.norm(dim=-1, keepdim=True)
        sim = torch.einsum('bte,bce->btc', x_feat, y_feat)
        sim = sim / self.temperature  # B,T,C
        return sim


class BinaryHead(nn.Module):
    def __init__(
            self,
            bin_head_type: str,
            d_model: int,
            num_layers: int = 3,
            kernel_size: int = 3,
            activation: str = 'gelu',
            norm: str = 'layernorm',
            adp_alpha_init: float = 0.0,
            adp_weight: float = 10.0,
    ):
        super().__init__()
        kwargs = {
            'd_model': d_model,
            'num_layers': num_layers,
            'kernel_size': kernel_size,
            'activation': activation,
            'norm': norm
        }
        self.adp_weight = adp_weight
        self.bin_head_type = bin_head_type
        if bin_head_type == 'vanilla':
            self.bin_head = ConvScoreHead(**kwargs)
        elif bin_head_type == 'fused_vanilla':
            self.bin_fused_head = ConvScoreHead(**kwargs)
        elif bin_head_type == 'adaptive':
            self.bin_head = ConvScoreHead(**kwargs)
            self.bin_fused_head = ConvScoreHead(**kwargs)
            self.adp_alpha = nn.Parameter(torch.zeros(1).fill_(adp_alpha_init))
        else:
            raise NotImplementedError

    def forward(self, before_fused: Tensor = None, after_fused: Tensor = None):
        # input [B,T,C]
        assert before_fused is not None or after_fused is not None

        out_logits = None
        if before_fused is not None and after_fused is None:  # vanilla
            out_logits = self.bin_head(before_fused.permute(0, 2, 1)).squeeze(1)
        if before_fused is None and after_fused is not None:  # fused_vanilla
            out_logits = self.bin_fused_head(after_fused.permute(0, 2, 1)).squeeze(1)
        if before_fused is not None and after_fused is not None:  # adaptive
            w = torch.sigmoid(self.adp_alpha * self.adp_weight)
            out_logits = self.bin_head(before_fused.permute(0, 2, 1)).squeeze(1) * w + \
                         self.bin_fused_head(after_fused.permute(0, 2, 1)).squeeze(1) * (1 - w)

        return out_logits


class MultiClassHead(nn.Module):
    def __init__(
            self,
            multi_head_type: str,
            d_model: int = 512,
            d_proj: int = 512,
            temperature_init: float = 0.2,
    ):
        super().__init__()
        if multi_head_type == 'sim':
            self.sim_head = SimScoreHead(temperature_init=temperature_init)
        elif multi_head_type == 'proj_sim':
            self.sim_head = ProjSimScoreHead(d_model, d_proj, temperature_init=temperature_init)
        else:
            raise NotImplementedError

    def forward(self, x_feat: Tensor, y_feat: Tensor):
        return self.sim_head(x_feat, y_feat)

