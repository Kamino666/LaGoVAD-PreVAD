from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from os import PathLike
import numpy as np
import torch
import json
import re
import random
from typing import Optional, List, Union
from .utils import resample_or_pad_feature_length
from .utils import BinaryBalancedBatchSampler
from .base import BaseDataModule, BaseDataset


DEFAULT_CLASSES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism"
]


class UCFCrimeFeatureDataset(Dataset):
    def __init__(self,
                 feature_dir: PathLike,
                 list_file: PathLike,
                 gt_dir: PathLike = None,
                 cls_embed_file: PathLike = None,
                 max_len: int = 200,
                 test_mode: bool = False) -> None:
        """
        Args:
            feature_dir (PathLike):
                The directory containing numpy format feature files. Each file should be named as '<video_id>.npy'.
            list_file (PathLike):
                A text file that contains the names of videos in a split set.
            gt_dir (Optional[PathLike]):
                Path to a directory that contains ground truth files. Each file is a boolean NumPy array with
                shape [num_frames], where positive values represent anomaly frames.
            cls_embed_file (Optional[PathLike]):
                Path to a NumPy file with CLIP embedding vectors for anomaly categories. The file should have
                shape [num_categories, embedding_dim]. Used in Prompt Enhanced Learning.
            max_len (int):
                Max length of feature sequence.
            test_mode (bool):
                Whether in test mode. If it is, no resample or pad to the feature.
        """
        super().__init__()
        self.max_len = max_len
        self.test_mode = test_mode
        self.feat_dir = Path(feature_dir)
        self.video_names = open(list_file).read().strip().split()
        self.labels = [torch.zeros([1]) if 'Normal' in n else torch.ones([1])
                       for n in self.video_names]

        self.gt_dir = Path(gt_dir) if gt_dir is not None else None
        self.cls_embeddings = np.load(cls_embed_file) if cls_embed_file is not None else None

        self.cls2idx = {'Normal': 0, 'Abuse': 1, 'Arrest': 2, 'Arson': 3, 'Assault': 4,
                        'Burglary': 5, 'Explosion': 6, 'Fighting': 7, 'RoadAccidents': 8,
                        'Robbery': 9, 'Shooting': 10, 'Shoplifting': 11, 'Stealing': 12, 'Vandalism': 13}

    def __getitem__(self, item):
        """
        Args:
            item (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following items:

                - 'v_feat' (Tensor): Visual feature with shape [num_frm, d_feature].
                - 'v_feat_l' (Tensor): Length of the visual feature with shape [1].
                - 'label' (LongTensor): Video level label indicating normal (0) or anomaly (1) with shape [1].
                - 'multi_cls_label' (LongTensor): Video level multi-class label with shape [1].
                - 'frame_gt' (optional) (BoolTensor): Frame level ground truth anomaly information with shape [num_frm].
                - 'prompt_feat' (optional) (Tensor): Text feature with shape [2, d_feature] containing foreground and background.
        """
        # Load visual features
        visual_feat = torch.tensor(np.load(self.feat_dir / self.video_names[item].strip()))
        if self.test_mode:
            visual_feat_length = [len(visual_feat)]
        else:
            visual_feat, visual_feat_length = resample_or_pad_feature_length(visual_feat, target_length=self.max_len)
        visual_feat_length = torch.tensor(visual_feat_length)

        # Retrieve video level label
        label = self.labels[item]
        video_name = self.video_names[item].strip().split('/')[-1].split('_')[0]  # Example: Abuse028 Normal
        cls_name = video_name[:-3] if 'Normal' not in video_name else 'Normal'
        multi_cls_label = torch.tensor(self.cls2idx[cls_name])

        # Prepare the result dictionary
        res_dict = dict(v_feat=visual_feat, v_feat_l=visual_feat_length, label=label, multi_cls_label=multi_cls_label)

        # Ground truth frame-level annotations
        if self.gt_dir:
            if video_name == 'Normal':
                gt_name = Path(self.video_names[item]).stem
                gt_name = '_'.join(gt_name.split('_')[:4])
                frame_gt = np.load(str(self.gt_dir / f"{gt_name}__9_gt.npy"))
            else:
                frame_gt = np.load(str(self.gt_dir / f"{video_name}_x264__9_gt.npy"))
            frame_gt = torch.tensor(frame_gt)
            if not self.test_mode:
                frame_gt, _ = resample_or_pad_feature_length(frame_gt, target_length=self.max_len * 16)
            res_dict['frame_gt'] = frame_gt

        # Prompt features (optional)
        if self.cls_embeddings is not None:
            fg_feat = self.cls_embeddings[self.cls2idx[cls_name]]
            bg_feat = self.cls_embeddings[self.cls2idx['Normal']]
            prompt_feat = np.stack((bg_feat, fg_feat), axis=0)
            res_dict['prompt_feat'] = prompt_feat

        return res_dict

    def __len__(self):
        return len(self.video_names)


class UCFCrimePromptedTestDataset(Dataset):
    def __init__(self,
                 visual_feature_dir,
                 text_feature_dir,
                 gt_path,
                 txt_max_len: int = 77,
                 return_one_query: bool = True
                 ) -> None:
        self.visual_feat_dir = Path(visual_feature_dir)
        self.text_feature_dir = Path(text_feature_dir)
        with open(gt_path) as f:
            self.gts = json.load(f)  # ground truths

        self.txt_max_len = txt_max_len
        self.return_one_query = return_one_query

        self.label2idx = {
            'anomaly': 1,
            'normal': 0
        }

    def __getitem__(self, item):
        # Load visual features
        vid_path = self.gts[item]['path']
        visual_feat = np.load(self.visual_feat_dir / vid_path)  # T,C
        visual_feat_length = torch.tensor([len(visual_feat)])

        # Load Text features
        if self.return_one_query and type(self.gts[item]['text']) is list:
            text_query = self.gts[item]['text'][0]
            text_path = self.gts[item]['text_path']
            text_feat = torch.tensor(np.load(self.text_feature_dir / text_path))[0]
            text_feature_length = (text_feat.mean(-1) != 0).sum(-1)
        else:
            text_query = self.gts[item]['text']
            text_path = self.gts[item]['text_path']
            text_feat = torch.tensor(np.load(self.text_feature_dir / text_path))  # B,T,C
            text_feature_length = (text_feat.mean(-1) != 0).sum(-1)

        # Retrieve video label
        label = self.gts[item]['label']
        target = torch.tensor(self.label2idx[label])
        temp_anno = torch.tensor(self.gts[item]['temporal_annotation']).view(-1, 2)

        # Prepare the result dictionary
        res_dict = dict(v_feat=visual_feat, v_feat_l=visual_feat_length,
                        t_feat=text_feat, t_feat_l=text_feature_length,
                        label=label, target=target, text=text_query,
                        temp_anno=temp_anno, video_path=vid_path)
        return res_dict

    def __len__(self):
        return len(self.gts)


class UCFCrimeOVDataset(Dataset):
    def __init__(self,
                 visual_feature_dir,
                 gt_path,
                 label_path,
                 vis_max_len: int = 512,
                 ) -> None:
        self.visual_feat_dir = Path(visual_feature_dir)
        self.vis_max_len = vis_max_len
        with open(gt_path) as f:
            self.gts = json.load(f)  # ground truths
        with open(label_path) as f:
            label_info = json.load(f)
            self.labels: list = label_info['labels']

    def __getitem__(self, item):
        # Load visual features
        vid_path = self.gts[item]['path']
        visual_feat = torch.tensor(np.load(self.visual_feat_dir / vid_path), dtype=torch.float32)  # T,C
        visual_feat_real_length = torch.tensor([len(visual_feat)])
        visual_feat, visual_feat_length = resample_or_pad_feature_length(visual_feat, target_length=self.vis_max_len)

        # Load Text features
        label_text = self.gts[item]['path'].replace('_x264.npy', '')
        label_text = re.sub(r'\d+', '', label_text)
        if 'Normal' in label_text:
            label_text = 'Normal'
        index_cls = torch.tensor(self.labels.index(label_text))

        temp_anno = torch.tensor(self.gts[item]['temporal_annotation']).view(-1, 2)  # in real frame

        # Prepare the result dictionary
        res_dict = dict(v_feat=visual_feat, v_feat_l=visual_feat_length,
                        label=index_cls, labels=[i.lower() for i in self.labels],
                        target_length=visual_feat_real_length,
                        temp_anno=temp_anno, video_path=vid_path)
        return res_dict

    def __len__(self):
        return len(self.gts)


# class UCFCrimeDataset(Dataset):
#     def __init__(self,
#                  data_root,
#                  gt_json_path,
#                  vis_max_len: int = 512,
#                  class_names: Optional[List[str]] = None
#                  ) -> None:
#         self.abbr = 'ucf'
#         self.root = Path(data_root)
#         with open(gt_json_path) as f:
#             self.gts = json.load(f)
#
#         self.vis_max_len = vis_max_len
#         self.max_span_num = 4
#
#         if class_names is not None:
#             self.class_names = class_names
#         else:
#             # self.class_names = list(set([i['class_name'] for i in self.gts]))
#             # self.class_names.sort()
#             self.class_names = DEFAULT_CLASSES.copy()
#
#         self.num_overlen_clips = 0
#
#     def __getitem__(self, item):
#         # Load visual features
#         feat_path = self.gts[item]['path']
#         vid_path = str(self.root / self.gts[item]['video_path'])
#         vid = feat_path.replace('.npy', '')
#         visual_feat = torch.tensor(np.load(self.root / feat_path), dtype=torch.float32)
#         if len(visual_feat) > self.vis_max_len:
#             self.num_overlen_clips += 1
#         visual_feat, visual_feat_length = resample_or_pad_feature_length(
#             visual_feat, target_length=self.vis_max_len
#         )
#         visual_feat_length = torch.tensor(visual_feat_length)
#
#         # Load Text
#         class_name = self.gts[item]['class_name']
#         class_idx = torch.tensor(self.class_names.index(class_name))
#
#         # Load Description
#         descs = self.gts[item].get('descriptions', None)
#         if type(descs) is list:
#             desc = random.choice(descs)
#         else:
#             desc = ""
#
#         # get temporal pseudo annotation
#         normed_span = torch.tensor(self.gts[item]['anomaly_span'], dtype=torch.float)
#         if normed_span.shape[0] == 0:  # Normal video
#             # print(self.gts[item])
#             frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
#             span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
#             span_ts *= -1
#         else:
#             normed_span = (normed_span * visual_feat_length).long()
#             frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
#             for i in range(len(normed_span)):
#                 frame_label[normed_span[i, 0]:normed_span[i, 1]] = 1.0
#             span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
#             span_ts *= -1
#             span_ts[:len(normed_span), :] = normed_span
#
#         # Prepare the result dictionary
#         res_dict = dict(
#             v_feat=visual_feat, v_feat_l=visual_feat_length, target_length=visual_feat_length,
#             cls_label=class_name, cls_label_idx=class_idx,
#             desc_label=desc,
#             video_path=vid_path,
#             pseudo_frame_label=frame_label,
#             pseudo_span=span_ts,
#             temp_span=span_ts  # only for evaluation
#         )
#
#         return res_dict
#
#     def __len__(self):
#         return len(self.gts)


class AnomalyVideoDataModule(BaseDataModule):
    def __init__(self,
                 data_root: str,
                 train_batch_size=16,
                 val_batch_size=16,
                 feature_name: str = 'I3D',
                 max_len: int = 200,
                 balance_sample: bool = False
                 ):
        super().__init__()
        self.test_set = None
        self.val_set = None
        self.train_set = None

        self.data_root = Path(data_root)
        self.feature_name = feature_name
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.balance_sample = balance_sample

        self.abnormal_dict = {'Normal': 0, 'Abuse': 1, 'Arrest': 2, 'Arson': 3, 'Assault': 4,
                              'Burglary': 5, 'Explosion': 6, 'Fighting': 7, 'RoadAccidents': 8,
                              'Robbery': 9, 'Shooting': 10, 'Shoplifting': 11, 'Stealing': 12,
                              'Vandalism': 13}

    def prepare_data(self) -> None:
        print("Please download original videos from https://www.crcv.ucf.edu/projects/real-world/")

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set = UCFCrimeFeatureDataset(
                self.data_root / self.feature_name,
                self.data_root / 'train.list',
                cls_embed_file=self.data_root / 'ucf-prompt.npy',
                max_len=self.max_len,
                test_mode=False,
            )
        if stage in ['fit', 'validation']:
            self.val_set = UCFCrimeFeatureDataset(
                self.data_root / self.feature_name,
                self.data_root / 'test.list',
                gt_dir=self.data_root / 'single_gt',
                cls_embed_file=self.data_root / 'ucf-prompt.npy',
                max_len=self.max_len,
                test_mode=False,
            )
        if stage == 'test':
            self.test_set = UCFCrimeFeatureDataset(
                self.data_root / self.feature_name,
                self.data_root / 'test.list',
                gt_dir=self.data_root / 'single_gt',
                cls_embed_file=self.data_root / 'ucf-prompt.npy',
                max_len=self.max_len,
                test_mode=True,
            )

    def train_dataloader(self) -> DataLoader:
        if self.balance_sample is True:
            return DataLoader(
                self.train_set, num_workers=16, pin_memory=True,
                batch_sampler=BinaryBalancedBatchSampler(
                    self.train_set, batch_size=self.train_batch_size,
                    cls_ratio=0.5, shuffle=True
                )
            )
        else:
            return DataLoader(self.train_set, batch_size=self.train_batch_size,
                              shuffle=True, num_workers=16, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        # return DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=8)
        return DataLoader(self.val_set, batch_size=self.val_batch_size,
                          shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=8)


class UCFCrimeDataset(BaseDataset):
    def __init__(self,
                 data_root,
                 gt_json_path,
                 vis_max_len: int = 512,
                 class_names: Optional[List[str]] = None,
                 ) -> None:
        super().__init__(
            data_root, gt_json_path,
            vis_max_len=vis_max_len,
            class_names=DEFAULT_CLASSES.copy() if class_names is None else class_names,
            max_span_num=4
        )
        self.abbr = 'ucf'
