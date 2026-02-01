import torch
import torch.nn as nn


def smooth_reg_loss(logits, lengths):
    """
    From `Real-world Anomaly Detection in Surveillance Videos`
    Args:
        logits: Tensor [B,...,T]
        lengths: Tensor [B]
    """
    bs = logits.shape[0]
    logits = logits.sigmoid()  # [B,...,T]
    diff = (logits[..., 1:] - logits[..., :-1]) ** 2  # [B,...,T-1]

    loss = 0
    for bi in range(bs):
        loss = loss + diff[bi, ..., :lengths[bi] - 1].mean()
    loss = loss / bs
    return loss


def sparsity_reg_loss(logits, lengths):
    """
    From `Real-world Anomaly Detection in Surveillance Videos`
    Args:
        logits: Tensor [B,...,T]
        lengths: Tensor [B]
    """
    bs = logits.shape[0]
    logits = logits.sigmoid()  # [B,...,T]

    loss = 0
    for bi in range(bs):
        loss = loss + logits[bi, ..., :lengths[bi]].mean()
    loss = loss / bs
    return loss


def mil_loss(logits, labels, lengths, topk_num: int = None, topk_pct: int = None):
    """
    vanilla multiple instance learning loss
    Args:
        topk_num: number of top-K instances
        topk_pct: topk_num = length / topk_pct
        logits: Tensor [B,T]
        labels: Tensor [B] binary labels
        lengths: Tensor [B]
    Returns:

    """
    bs = logits.shape[0]
    assert topk_num is not None or topk_pct is not None

    running_logits = []
    for bi in range(bs):
        if topk_num is None:
            k = int(max((lengths[bi] // topk_pct).cpu().item(), 1))
        else:
            k = int(max(topk_num, 1))
        top_values, top_indices = torch.topk(logits[bi, :lengths[bi]], k=int(k), dim=0)  # k,C
        running_logits.append(top_values.mean())
    running_logits = torch.stack(running_logits, dim=0)
    loss = nn.BCEWithLogitsLoss()(running_logits, labels)
    return loss


def multi_class_mil_loss(logits, labels, lengths, topk_num: int = None, topk_pct: int = None):
    """
    select top-K instances for each class, aggregate them as the video-level logits,
    and apply cross entropy loss using GT labels.
    Args:
        logits: Tensor [B,T,C]
        labels: Tensor [B] including normal class
        lengths: Tensor [B]
        topk_num: number of top-K instances
        topk_pct: topk_num = length / topk_pct
    Returns:
    """
    bs = logits.shape[0]
    assert topk_num is not None or topk_pct is not None

    running_logits = []
    for bi in range(bs):
        if topk_num is None:
            k = int(max((lengths[bi] // topk_pct).cpu().item(), 1))
        else:
            k = int(max(topk_num, 1))
        top_values, top_indices = torch.topk(logits[bi, :lengths[bi]], k=int(k), dim=0)  # k,C
        running_logits.append(top_values.mean(dim=0))  # C
    running_logits = torch.stack(running_logits, dim=0)  # B,C
    loss = nn.CrossEntropyLoss()(running_logits, labels)
    return loss


def multi_class_mil_loss_v2(logits, labels, lengths, topk_num: int = None, topk_pct: int = None):
    """
    select top-K instances for each class, aggregate them as the video-level logits,
    and apply cross entropy loss using GT labels.
    Args:
        logits: Tensor [B,T,C]
        labels: Tensor [B] including normal class
        lengths: Tensor [B]
        topk_num: number of top-K instances
        topk_pct: topk_num = length / topk_pct
    Returns:
    """
    bs = logits.shape[0]
    assert topk_num is not None or topk_pct is not None

    running_logits = []
    for bi in range(bs):
        if topk_num is None:
            k = int(max((lengths[bi] // topk_pct).cpu().item(), 1))
        else:
            k = int(max(topk_num, 1))
        if labels[bi] == 0:  # For normal, use bottom-K as global class probability
            top_values, top_indices = torch.topk(logits[bi, :lengths[bi]], k=int(k), dim=0, largest=False)  # k,C
        else:
            top_values, top_indices = torch.topk(logits[bi, :lengths[bi]], k=int(k), dim=0)  # k,C
        running_logits.append(top_values.mean(dim=0))  # C
    running_logits = torch.stack(running_logits, dim=0)  # B,C
    loss = nn.CrossEntropyLoss()(running_logits, labels)
    return loss


def supervised_loss(logits, frame_labels, lengths):
    """
    vanilla supervised loss
    Args:
        logits: Tensor [B,T]
        frame_labels: Tensor [B,T] binary labels
        lengths: Tensor [B]
    Returns:

    """
    mask = (torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]).float()  # [B,T]
    loss = nn.BCEWithLogitsLoss(reduction='none')(logits, frame_labels.float())
    loss = loss * mask
    loss = loss.sum() / mask.sum()
    return loss


def pseudo_sup_mil_loss(logits, frame_labels, lengths, topk_pct: int = None):
    """
    multiple instance learning loss that select top-K in annotated span
    Args:
        logits: Tensor [B,T]
        frame_labels: Tensor [B,T] binary labels
        lengths: Tensor [B]
        topk_pct: topk_num = length / topk_pct
    Returns:

    """
    bs = logits.shape[0]
    frame_labels_mask = torch.zeros_like(frame_labels)
    frame_labels_mask[frame_labels == 0] = -100

    running_logits, running_labels = [], []
    for bi in range(bs):
        if frame_labels[bi].sum() == 0:  # Normal
            k = int(max((lengths[bi] // topk_pct).cpu().item(), 1))
            top_values, top_indices = torch.topk(logits[bi, :lengths[bi]], k=int(k), dim=0)  # k
            running_labels.append(0)
        else:  # Abnormal
            k = int(max((frame_labels[bi].sum() // topk_pct).cpu().item(), 1))
            top_values, top_indices = torch.topk(logits[bi, :] + frame_labels_mask[bi, :], k=int(k), dim=0)
            running_labels.append(1)
        running_logits.append(top_values.mean())
    running_logits = torch.stack(running_logits, dim=0)
    running_labels = torch.tensor(running_labels, device=running_logits.device, dtype=running_logits.dtype)
    loss = nn.BCEWithLogitsLoss()(running_logits, running_labels)
    return loss


def asymmetric_infonce_loss(sim, label):
    """

    Args:
        sim: Tensor SxB, S is the number of anomaly, B is batchsize
        label: Tensor S, indicate the index of anomaly in the batch

    Returns:

    """
    # text --> video
    loss_t2v = nn.CrossEntropyLoss(label_smoothing=0.1)(sim, label)

    # video --> text
    ano_sim = sim[:, label]  # S,S
    loss_v2t = nn.CrossEntropyLoss(label_smoothing=0.1)(ano_sim, torch.arange(ano_sim.shape[0], device=ano_sim.device))

    loss = (loss_t2v + loss_v2t) / 2
    return loss


class CapContrastLoss(nn.Module):
    def __init__(self, contrast_type: str, temperature=0.02):
        super().__init__()
        # vanilla: no negative mining
        # n3: Select the normal portion of the abnormal video that is not completely abnormal as the NEG
        # n1: inverted aggregated video feature as neg example
        # n2: avg aggregated video feature as neg example
        self.contrast_type = contrast_type
        self.t = temperature

    def forward(
            self,
            logits: torch.Tensor,
            lengths: torch.Tensor,
            v_feats: torch.Tensor,
            t_feats: torch.Tensor,
            cls_label_idx: torch.Tensor,
            pseudo_frame_label: torch.Tensor,
    ):
        """

        Args:
            logits: [B,T] Binary logits before softmax or sigmoid
            lengths: [B] Lengths of each video
            v_feats: [B,T,E] Video features
            t_feats: [S,E] Text features (excluding Normal)
            cls_label_idx: [B] 0=Normal, others=Abnormal
            pseudo_frame_label: [B,T] fine-grained frame-level label

        Returns:

        """
        max_len = logits.shape[1]
        device = logits.device
        bs = logits.shape[0]
        ano_indices = torch.where(cls_label_idx != 0)[0]  # S

        # Vanilla Contrastive
        # make mask L=5: 0 0 0 0 0 -1e4 -1e4 -1e4
        mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
        mask = torch.where(mask == True, -1e4, 0)
        masked_logits = logits + mask  # [B,T]
        masked_attn = torch.softmax(masked_logits / self.t, dim=1).unsqueeze(1)  # [B,1,T]
        agg_v_feats = (masked_attn @ v_feats).squeeze(1)  # [B,E]

        # N3 Neg mining
        if self.contrast_type == 'n3':
            mining_v_feats = []
            for i in range(bs):
                if i not in ano_indices:  # only anomaly video
                    continue
                ano_logit = logits[i, :lengths[i]]  # [T]
                ano_prob = ano_logit.sigmoid()  # [T]
                if ano_prob.max() - ano_prob.min() < 0.2:  # only video that is not completely abnormal
                    # print(f"Skip due to max-min={ano_prob.max() - ano_prob.min()}<0.2")
                    continue
                # select normal portion
                mining_logit = ano_logit[pseudo_frame_label[i, :lengths[i]] != 0]  # [T']
                mining_attn = torch.softmax(-mining_logit / self.t, dim=0)  # [T']
                mining_v_feats.append(
                    mining_attn @ v_feats[i][pseudo_frame_label[i] != 0]
                )  # E
            # print(f"Mining {len(mining_v_feats)} samples from {len(ano_indices)} anomaly videos")
            if len(mining_v_feats) > 0:
                mining_v_feats = torch.stack(mining_v_feats, dim=0)  # [S',E]
                agg_v_feats = torch.cat([agg_v_feats, mining_v_feats], dim=0)

        # compute asymmetric_infonce_loss
        sim_mat = torch.einsum('se,be->sb', t_feats, agg_v_feats)
        loss = asymmetric_infonce_loss(sim_mat, ano_indices)
        return loss
