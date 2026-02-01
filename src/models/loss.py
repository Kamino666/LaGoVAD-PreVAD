import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_mil_loss(scores, targets: float, topk=5):
    # B,L,?
    scores = scores.view(scores.shape[0], scores.shape[1])  # B,L
    top_scores = scores.topk(topk, dim=1)[0].mean(-1)  # B,
    targets = torch.ones_like(top_scores) * targets
    loss = nn.BCEWithLogitsLoss(reduction='sum')(top_scores, targets)
    return loss


def thresh_mil_loss(scores, targets: float, thresh: float = 0.5):
    # B,L,?
    scores = scores.view(scores.shape[0], scores.shape[1])  # B,L
    mask = scores.sigmoid() > thresh  # B,L
    top_scores = torch.sum(scores * mask, dim=-1) / (torch.sum(mask, dim=-1) + 1e-6)  # B,
    targets = torch.ones_like(top_scores) * targets
    loss = nn.BCEWithLogitsLoss(reduction='sum')(top_scores, targets)
    return loss


def oic_loss(
        scores: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor,
        inflation_rate: float = 0.25,
):
    # scores: B,L,1   labels: B   coords: B,L,2
    abn_scores = scores[labels == 1]  # S,L,1
    abn_coords = coords[labels == 1]  # S,L,2
    top_idx = abn_scores.argmax(dim=1).squeeze(-1)  # S
    top_abn_coord = abn_coords.gather(1, top_idx[:, None, None].expand(-1, -1, 2))  # S,1,2
    inflated_coord = top_abn_coord * (1 + inflation_rate)
    top_abn_coord[:, :, 0] *= -1
    inflated_coord[:, :, 0] *= -1
    inner_coord = top_abn_coord + top_idx[:, None, None]  # S,1,2
    outer_coord = inflated_coord + top_idx[:, None, None]  # S,1,2
    inner_coord = inner_coord.clamp(0, scores.shape[1])
    outer_coord = outer_coord.clamp(0, scores.shape[1])
    oic = []
    for i in range(inner_coord.shape[0]):
        c = inner_coord[i, :].long()
        inner_score = abn_scores[i, c[0]: c[1]].sum() / (inner_coord[i, :, 1] - inner_coord[i, :, 0])
        c = outer_coord[i, :].long()
        outer_score = abn_scores[i, c[0]: c[1]].sum() / (outer_coord[i, :, 1] - outer_coord[i, :, 0])
        oic.append(outer_score - inner_score)
    oic = torch.stack(oic, dim=0).mean()
    return oic


# IVC loss
# [AAAI22] Weakly Supervised Video Moment Localization with Contrastive Negative Sample Mining
def intra_video_contrastive_loss(scores: torch.Tensor,
                                 hidden_states: torch.Tensor,
                                 masks: torch.Tensor):
    # scores: [B,L]  hidden_states: [B,L,C]  masks[B,L]
    attention_mask = masks.to(scores.dtype)
    attention_mask = (1.0 - attention_mask) * torch.finfo(scores.dtype).min

    # all [B,L]
    pos_w = torch.softmax(scores + attention_mask, dim=-1)
    easy_neg_w = 1 - pos_w
    hard_neg_w = masks.float() / masks.sum(dim=1, keepdim=True)




