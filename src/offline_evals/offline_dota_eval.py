from tqdm import tqdm
import torchmetrics
import torch
import pickle
from pathlib import Path

from src.datasets.DoTA import DoTADataset


vis_dataset_index = 5
pred_path = "ckpts/pred_results.pkl"
with open(pred_path, "rb") as f:
    preds = pickle.load(f)[vis_dataset_index]
metric = torchmetrics.AUROC('binary')
dataset = DoTADataset(
    '/data/datasets/PreVAD',
    '/data/datasets/PreVAD/other_datasets/dota_test_anno.json',
    vis_max_len=512,
)
for i in tqdm(range(len(dataset))):
    ds_item, pred = dataset[i], preds[i]
    frame_label = ds_item['pseudo_frame_label'][:ds_item['target_length']]
    pred_score = torch.tensor(pred['score'])  # L
    pred_normed_score = (pred_score - pred_score.min()) / (pred_score.max() - pred_score.min())

    metric.update(pred_normed_score, frame_label)
print(f"AUC: {metric.compute()}")