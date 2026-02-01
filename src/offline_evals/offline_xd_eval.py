from tqdm import tqdm
import torchmetrics
import torch
import pickle
from pathlib import Path

from src.datasets.XDViolence import XDDataset


vis_dataset_index = 1
pred_path = "ckpts/pred_results.pkl"
with open(pred_path, "rb") as f:
    preds = pickle.load(f)[vis_dataset_index]
full_metric = torchmetrics.Accuracy('multiclass', num_classes=7)
f1_metric = torchmetrics.F1Score('multiclass', num_classes=7, average='macro')
dataset = XDDataset(
    '/data/datasets/PreVAD',
    '/data/datasets/PreVAD/other_datasets/xd_test_anno.json',
    vis_max_len=768,
)
for i in tqdm(range(len(dataset))):
    ds_item, pred = dataset[i], preds[i]

    pred_mul_score = torch.tensor(pred['real_mul_scores'])  # C,L
    cls_logits = torch.cat(
        [
            torch.min(pred_mul_score[0])[None,],
            torch.max(pred_mul_score[1:], dim=1)[0]
        ],
        dim=0
    )
    cls_probs = torch.softmax(cls_logits, dim=0)
    full_metric.update(cls_probs[None, :], ds_item['cls_label_idx'].unsqueeze(0))
    f1_metric.update(cls_probs[None, :], ds_item['cls_label_idx'].unsqueeze(0))

print(f"Acc: {full_metric.compute():.4f}")
print(f"F1: {f1_metric.compute():.4f}")
