import torch
import yaml
import argparse
import importlib
import subprocess
import math
import numpy as np
from pathlib import Path
from rich.progress import Progress
from datetime import datetime
import torchmetrics
from src.models.LaGoVAD.verbalizer import DatasetSpecVerbalizer
import src.datasets.UCFCrime
import src.datasets.XDViolence
import src.datasets.XDViolence
import src.datasets.MSAD
import src.datasets.LAD
import src.datasets.DoTA
import src.datasets.UBNormal
import src.datasets.TAD
import src.datasets.PreVAD
from lightning import seed_everything


seed_everything(7)
device = torch.device('cuda')

def extract_frames(video_path: str, output_dir: str, interval=8):
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', f'select="not(mod(n\,{interval}))"',
        '-vsync', 'vfr', '-q:v', '2',
        f'"{output_dir}/%05d.jpg"'
    ]
    try:
        print(' '.join(cmd))
        subprocess.run(' '.join(cmd), shell=True, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        print(e.stderr)
        return None
    output_paths = [str(p) for p in sorted(Path(output_dir).glob('*.jpg'))]
    print(f"Extracted {len(output_paths)} frames to {output_dir}")
    return output_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='ucf')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./tmp')
    args = parser.parse_args()
    return args


def build_model(cfg, ckpt_path):
    # build model
    model_cls_paths = cfg['model']['class_path'].split('.')
    model_cls = getattr(
        importlib.import_module('.'.join(model_cls_paths[:-1])),
        model_cls_paths[-1]
    )
    model_cfg_paths = cfg['model']['init_args']['model_config']['class_path'].split('.')
    model_cfg_cls = getattr(
        importlib.import_module('.'.join(model_cfg_paths[:-1])),
        model_cfg_paths[-1]
    )
    model_cfg = model_cfg_cls(**cfg['model']['init_args']['model_config']['init_args'])
    training_cfg_paths = cfg['model']['init_args']['training_config']['class_path'].split('.')
    training_cfg_cls = getattr(
        importlib.import_module('.'.join(training_cfg_paths[:-1])),
        training_cfg_paths[-1]
    )
    training_cfg = training_cfg_cls(**cfg['model']['init_args']['training_config']['init_args'])
    model = model_cls(model_cfg, training_cfg).to(device)

    # load checkpoints
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)['state_dict']
    load_states = model.load_state_dict(state_dict, strict=False)
    if len(load_states[1]) > 0:
        print(f"Load may failed")
        print(f"Unexpected Keys: {load_states[1]}")
    if not all([k for k in load_states[0] if k.startswith('clip_text_model.')]):
        print(f"Load may failed")
        print(f"Missing Keys: {load_states[0]}")

    model.eval()

    return model


def build_dataset(name='ubif', data_root='datasets/PreVAD'):
    data_root = Path(data_root)
    verbalizer = DatasetSpecVerbalizer()
    if name == 'ucf':
        dataset = src.datasets.UCFCrime.UCFCrimeDataset(
            data_root,
            data_root / 'other_datasets' / 'ucf_test_anno.json',
        )
        classes = src.datasets.UCFCrime.DEFAULT_CLASSES
    elif name == 'xd':
        dataset = src.datasets.XDViolence.XDDataset(
            data_root,
            data_root / 'other_datasets' / 'xd_test_anno.json',
        )
        classes = src.datasets.XDViolence.DEFAULT_CLASSES
    elif name == 'msad':
        dataset = src.datasets.MSAD.MSADDataset(
            data_root,
            data_root / 'other_datasets' / 'msad_test_anno.json',
        )
        classes = src.datasets.MSAD.DEFAULT_CLASSES
    elif name == 'lad':
        dataset = src.datasets.LAD.LADDataset(
            data_root,
            data_root / 'other_datasets' / 'lad_test_anno.json',
        )
        classes = src.datasets.LAD.DEFAULT_CLASSES
    elif name == 'dota':
        dataset = src.datasets.DoTA.DoTADataset(
            data_root,
            data_root / 'other_datasets' / 'dota_test_anno.json',
        )
        classes = src.datasets.DoTA.DEFAULT_CLASSES
    elif name == 'ubn':
        dataset = src.datasets.UBNormal.UBNormalDataset(
            data_root,
            data_root / 'other_datasets' / 'ubnormal_test_anno.json',
        )
        classes = src.datasets.UBNormal.DEFAULT_CLASSES
    elif name == 'tad':
        dataset = src.datasets.TAD.TADDataset(
            data_root,
            data_root / 'other_datasets' / 'tad_test_anno.json',
        )
        classes = src.datasets.TAD.DEFAULT_CLASSES
    elif name == 'prevad':
        dataset = src.datasets.PreVAD.PreVADDataset(
            data_root,
            data_root / 'v6' / 'prevad_test_anno_v6.json',
        )
        classes = src.datasets.PreVAD.DEFAULT_CLASSES
    else:
        raise NotImplementedError
    verbalizer.set_dataset(dataset.abbr)
    return dataset, verbalizer, classes


@torch.no_grad()
def main():
    max_vis_len = 512

    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        cfg['model']['init_args']['model_config']['init_args']['verbalizer_type'] = None
    # cache_dir = Path(args.cache_dir, datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S'))
    # cache_dir.mkdir()

    # build model
    model = build_model(cfg, args.ckpt)

    # build dataset
    dataset, verbalizer, classes = build_dataset(args.dataset, args.data_root)

    # inference loop
    _gts, _preds, _vid_names = [], [], []
    auroc_met = torchmetrics.AUROC(task='binary')
    ap_met = torchmetrics.AveragePrecision(task='binary')
    with Progress() as progress:
        dataset_task = progress.add_task('Inferencing', total=len(dataset))
        for i in range(len(dataset)):
            data = dataset.get_full_item(i)
            num_iter = math.ceil(data['v_feat_l'].item() / max_vis_len)
            vid_task = progress.add_task(Path(data['video_path']).stem, total=num_iter)

            # inference video loop
            vid_bin_scores, vid_mul_scores = [], []
            for j in range(num_iter):
                v_feats = data['v_feat'][j*max_vis_len: (j+1)*max_vis_len].unsqueeze(0)  # 1,T,E

                inputs_dict = {
                    'batch': {
                        'v_feat': v_feats.to(device),
                        'v_feat_l': torch.tensor([v_feats.shape[1]], device=device),
                    },
                    # 'class_names': list(ucf_cls_defs.values()),
                    # 'class_names': classes,
                    'class_names': verbalizer(classes),
                    # 'query_captions': verbalizer(classes),
                    # 'query_captions': list(ucf_cls_defs.values()),
                }
                outputs_dict = model(**inputs_dict)
                bin_score = outputs_dict['cls_bin_logits'][0].sigmoid().cpu()  # [T]
                mul_score = outputs_dict['cls_sim_mat'][0].sigmoid().cpu().T  # [S,T]
                # bin_score = outputs_dict['cap_bin_logits'][0].sigmoid().cpu()  # [T]
                # mul_score = outputs_dict['cap_sim_mat'][0].sigmoid().cpu().T  # [S,T]

                vid_bin_scores.append(bin_score)
                vid_mul_scores.append(mul_score)

                progress.update(vid_task, advance=1)
            vid_bin_scores = torch.concatenate(vid_bin_scores, dim=0)[:data['v_feat_l']]  # [T]
            vid_mul_scores = torch.concatenate(vid_mul_scores, dim=1)[:data['v_feat_l']]  # [S,T]

            # make gt
            gt = torch.zeros_like(vid_bin_scores)
            for span in data['temp_span']:  # N,2
                if span[0] == span[1]:
                    continue
                gt[span[0]:span[1]] = 1.0

            # update metric
            auroc_met.update(vid_bin_scores, gt.long())
            ap_met.update(vid_bin_scores, gt.long())

            _gts.append(np.repeat(gt.cpu().numpy(), 16))
            _preds.append(np.repeat(vid_bin_scores.cpu().numpy(), 16))
            _vid_names.append(Path(data['video_path']).stem)

            progress.update(vid_task, visible=False)
            progress.update(dataset_task, advance=1)

    # calculate metric
    auroc = auroc_met.compute()
    ap = ap_met.compute()

    print(f"AUC: {auroc:.4f}")
    print(f"AP: {ap:.4f}")


if __name__ == '__main__':
    main()



