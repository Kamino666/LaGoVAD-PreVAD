from lightning import LightningDataModule
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
import json
import random
from .utils import resample_or_pad_feature_length
from typing import Optional, List


class BaseDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

class BaseDataset(Dataset):
    def __init__(self,
                 data_root,
                 gt_json_path,
                 vis_max_len: int = 512,
                 class_names: Optional[List[str]] = None,
                 max_span_num: int = 25,
                 stream_mode=False,
                 ) -> None:
        self.abbr = '_base_'
        self.root = Path(data_root)
        with open(gt_json_path) as f:
            self.gts = json.load(f)

        self.vis_max_len = vis_max_len
        self.max_span_num = max_span_num
        self.class_names = class_names
        self.num_overlen_clips = 0
        self.stream_mode = stream_mode

    def get_full_item(self, item):
        # Load visual features
        feat_path = self.gts[item]['path']
        vid_path = str(self.root / self.gts[item]['video_path'])
        vid = feat_path.replace('.npy', '')
        visual_feat = torch.tensor(np.load(self.root / feat_path), dtype=torch.float32)  # T,E
        visual_feat_length = torch.tensor(len(visual_feat))

        # Load Text
        class_name = self.gts[item]['class_name']
        # class_idx = torch.tensor(self.class_names.index(class_name))
        class_idx = torch.tensor(-1)

        # Load Description
        descs = self.gts[item].get('descriptions', None)
        if type(descs) is list:
            desc = random.choice(descs)
        else:
            desc = ""

        # get temporal pseudo annotation
        normed_span = torch.tensor(self.gts[item]['anomaly_span'], dtype=torch.float)
        if normed_span.shape[0] == 0:  # Normal video
            # print(self.gts[item])
            frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
            span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
            span_ts *= -1
        else:
            normed_span = normed_span * visual_feat_length
            normed_span = torch.round(normed_span).long()

            frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
            for i in range(len(normed_span)):
                frame_label[normed_span[i, 0]:normed_span[i, 1]] = 1.0
            span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
            span_ts *= -1
            span_ts[:len(normed_span), :] = normed_span

        # Prepare the result dictionary
        res_dict = dict(
            v_feat=visual_feat, v_feat_l=visual_feat_length, target_length=visual_feat_length,
            cls_label=class_name, cls_label_idx=class_idx,
            desc_label=desc,
            video_path=vid_path,
            pseudo_frame_label=frame_label,
            pseudo_span=span_ts,
            temp_span=span_ts  # only for evaluation
        )

        return res_dict

    def __getitem__(self, item):
        # Load visual features
        feat_path = self.gts[item]['path']
        vid_path = str(self.root / self.gts[item]['video_path'])
        vid = feat_path.replace('.npy', '')
        visual_feat = torch.tensor(np.load(self.root / feat_path), dtype=torch.float32)
        if len(visual_feat) > self.vis_max_len:
            self.num_overlen_clips += 1
        visual_feat, visual_feat_length = resample_or_pad_feature_length(
            visual_feat, target_length=self.vis_max_len
        )
        visual_feat_length = torch.tensor(visual_feat_length)

        # Load Text
        class_name = self.gts[item]['class_name']
        class_idx = torch.tensor(self.class_names.index(class_name))

        # Load Description
        descs = self.gts[item].get('descriptions', None)
        if type(descs) is list:
            desc = random.choice(descs)
        else:
            desc = ""

        # get temporal pseudo annotation
        normed_span = torch.tensor(self.gts[item]['anomaly_span'], dtype=torch.float)
        if normed_span.shape[0] == 0:  # Normal video
            # print(self.gts[item])
            frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
            span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
            span_ts *= -1
        else:
            # normed_span = (normed_span * visual_feat_length).long()
            normed_span = normed_span * visual_feat_length
            normed_span = torch.round(normed_span).long()

            frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
            for i in range(len(normed_span)):
                frame_label[normed_span[i, 0]:normed_span[i, 1]] = 1.0
            span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
            span_ts *= -1
            span_ts[:len(normed_span), :] = normed_span

        # Prepare the result dictionary
        res_dict = dict(
            v_feat=visual_feat, v_feat_l=visual_feat_length, target_length=visual_feat_length,
            cls_label=class_name, cls_label_idx=class_idx,
            desc_label=desc,
            video_path=vid_path,
            pseudo_frame_label=frame_label,
            pseudo_span=span_ts,
            temp_span=span_ts  # only for evaluation
        )

        return res_dict

    def __len__(self):
        return len(self.gts)
