import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertEncoder
from .modeling_vadclip import Transformer, GraphConvolution, DistanceAdj, QuickGELU
from .modeling_roformer import RoFormerEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=1000):
        super().__init__()
        # Create a long enough P
        pe = torch.zeros((1, max_len, hidden_size))
        indices = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) /
                   torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size))
        pe[:, :, 0::2] = torch.sin(indices)
        pe[:, :, 1::2] = torch.cos(indices)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class VanillaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pos_encoding = PositionalEncoding(config.hidden_size)
        self.encoder = BertEncoder(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
    ):
        hidden_states = self.pos_encoding(hidden_states)
        return self.encoder(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict
        )


class LGTAdapter(nn.Module):
    def __init__(self, config, attn_window=15):
        super().__init__()
        visual_width = config.hidden_size
        visual_layers = config.num_hidden_layers
        visual_head = config.num_attention_heads
        visual_length = 512
        self.visual_length = visual_length

        attn_mask = self.build_attention_mask(attn_window)
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=attn_mask
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(-10e4)
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=images.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def forward(
            self,
            hidden_states,
            lengths,
    ):
        # hidden_states: [B,T,C]
        # attention_mask: [B]
        # -> enc_hidden_states: [B,T,C]
        enc_hidden_states = self.encode_video(hidden_states, lengths)
        return enc_hidden_states
