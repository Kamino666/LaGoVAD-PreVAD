import torch
import pathlib
from src.utils.visualization import vis_result


def save_pred_result(pred_res: dict, save_name=None, save_dir='./tmp'):
    vid = pathlib.Path(pred_res['video_path']).stem
    save_name = vid if save_name is None else save_name
    if 'temp_span' in pred_res:
        gt_scores = pred_res['temp_span'].flatten()  # 1,n,2 -> n*2
        gt_scores = gt_scores[gt_scores != -1].reshape(-1, 2)
    else:
        gt_scores = None
    vis_result(
        pred_res['score'], gt_scores,
        vid_path=str(pred_res['video_path']).replace('.npy', '.mp4'),
        save_dir=save_dir,
        save_filename=save_name
    )
    save_path = pathlib.Path(save_dir, f'{save_name}.png')
    return save_path, vid


def get_attention_mask(feat_lengths, max_len, device=None):
    device = feat_lengths.device if device is None else device
    mask = torch.zeros(len(feat_lengths), max_len, device=device)
    for i, l in enumerate(feat_lengths):
        mask[i, :l] = 1
    return mask.bool()
