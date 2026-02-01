import json

from torch.utils.data import DataLoader, Dataset, default_collate
from pathlib import Path
import numpy as np
import torch
import random
from collections import defaultdict

from .utils import truncate_or_pad_feature_length, resample_or_pad_feature_length

from typing import Optional, List, Union

# DEFAULT_CLASSES = [
#     'Normal',
#     'Accident',
#     'AirAccident',
#     'AnimalAttackAnimal',
#     'AnimalAttackHuman',
#     'AnimalPredation',
#     'CarAccident',
#     'Collapse',
#     'CrowdViolence',
#     'Explosion',
#     'FallDown',
#     'FallIntoWater',
#     'Fire',
#     'MechanicalAccident',
#     'ObjectImpact',
#     'RangeShooting',
#     'Riot',
#     'Robbery',
#     'Shooting',
#     'TrainAccident',
#     'Violence',
#     'WarScene'
# ]
DEFAULT_CLASSES = [
    'Normal',
    'Vehicle Accident', 'Air Accident', 'Train Accident', 'Car Accident',
    'Violence', 'Vandalism', 'Crowd Violence', 'Riot', 'Assault', 'Range Shooting', 'Shooting Accident', 'War',
    'Robbery', 'Carjacking', 'Mugging', 'Store Robbery',
    'Production Accident', 'Mechanical Accident', 'Object Impact', 'Collapse', 'Fall from Height',
    'Fire-related Accident', 'Fume', 'Fire', 'Explosion',
    'Animal-related Violence', 'Predation', 'Animal Attack Animal', 'Animal Attack Human',
    'Daily Accident', 'Sport Fail', 'Stunt Fail', 'Fall into Water', 'Fall to the Ground', 'Drop Something'
]
"""
1. Vehicle Accident
    1.1. Air Accident
    1.2. Train Accident
    1.3. Car Accident
    1.4. Others
2. Violence
    2.1. Vandalism
    2.2. Crowd Violence
    2.3. Riot
    2.4. Assault
    2.5. Range Shooting
    2.6. Shooting Accident
    2.7. War
    2.8. Others
3. Robbery
    3.1. Carjacking
    3.2. Mugging
    3.3. Store Robbery
    3.4. Others
4. Production Accident
    4.1. Mechanical Accident
    4.2. Object Impact
    4.3. Collapse
    4.4. Fall from Height
    4.5. Others
5. Fire-related Accident
    5.1. Fume
    5.2. Fire
    5.3. Explosion
    5.4. Others                       # no
6. Animal-related Violence
    6.1. Predation
    6.2. Animal Abuse                 # no
    6.3. Animal Attack Animal
    6.4. Animal Attack Human
    6.5. Others
7. Daily Accident
    7.1. Sport Fail
    7.2. Stunt Fail
    7.3. Fall into Water
    7.4. Fall to the Ground
    7.5. Drop Something
    7.6. Others
"""


def default_collate_without_numpy(batch):
    # batch: list[dict]
    batch = default_collate(batch)
    batch['v_feat'] = batch['v_feat'].numpy()
    return batch


class PreVADFeatureDataset(Dataset):
    def __init__(self,
                 visual_feature_dir,
                 text_feature_dir,
                 gt_path,
                 pseudo_label_dir=None,
                 pseudo_label_type='logits',
                 vis_max_len: int = 128,
                 txt_max_len: int = 77,
                 test_mode: bool = False) -> None:
        self.visual_feat_dir = Path(visual_feature_dir)
        self.text_feature_dir = Path(text_feature_dir)
        with open(gt_path) as f:
            self.gts = json.load(f)  # ground truths

        self.pseudo_label_type = pseudo_label_type
        if pseudo_label_dir is not None:
            self.pseudo_label_dir = Path(pseudo_label_dir)
        else:
            self.pseudo_label_dir = None

        self.vis_max_len = vis_max_len
        self.txt_max_len = txt_max_len
        self.test_mode = test_mode

        self.label2idx = {
            'anomaly': 1,
            'normal': 0
        }

    def __getitem__(self, item):
        # Load visual features
        vid_path = self.gts[item]['path']
        vid = vid_path.replace('.npy', '')
        visual_feat = torch.tensor(np.load(self.visual_feat_dir / vid_path), dtype=torch.float32)
        if self.test_mode:
            visual_feat_length = [len(visual_feat)]
        else:
            visual_feat, visual_feat_length = truncate_or_pad_feature_length(visual_feat,
                                                                             target_length=self.vis_max_len)
        visual_feat_length = torch.tensor(visual_feat_length)

        # Load Text features
        text_query = self.gts[item]['text']
        text_path = self.gts[item]['text_path']
        text_feat = torch.tensor(np.load(self.text_feature_dir / text_path), dtype=torch.float32)
        text_feat, text_feature_length = truncate_or_pad_feature_length(text_feat, target_length=self.txt_max_len)
        text_feature_length = torch.tensor(text_feature_length)

        # Retrieve video label
        label = self.gts[item]['label']
        target = torch.tensor(self.label2idx[label])

        # Load Pseudo-label
        pseudo_label = None
        if self.pseudo_label_dir is not None:
            if self.pseudo_label_type in ['logits', 'saliency']:
                pseudo_label = torch.tensor(np.load(
                    self.pseudo_label_dir / f"{vid}_{self.pseudo_label_type}.npy"
                ))  # l
                pseudo_label = torch.nn.functional.interpolate(pseudo_label[None, None, :],
                                                               size=visual_feat_length,
                                                               mode='linear').flatten()  # l -> L
                pseudo_label = truncate_or_pad_feature_length(pseudo_label, target_length=self.vis_max_len)[0]
            elif self.pseudo_label_type == 'fusion_v1':
                ps_logits = torch.tensor(np.load(self.pseudo_label_dir / f"{vid}_logits.npy"))  # L
                ps_spans = torch.tensor(np.load(self.pseudo_label_dir / f"{vid}_span.npy"))  # n,2
                ps_top_span = ps_spans[ps_logits.argmax(0), :].long()  # 2
                ps_saliency = torch.tensor(np.load(self.pseudo_label_dir / f"{vid}_saliency.npy"))  # L
                ps_max_saliency_idx = ps_saliency.argmax()  # 1
                pseudo_label = torch.zeros(len(ps_logits), dtype=torch.float)
                pseudo_label[ps_top_span[0]:ps_top_span[1]] = 0.8
                pseudo_label[max(ps_max_saliency_idx - 2, 0): ps_max_saliency_idx + 3] = 1.0
                pseudo_label = torch.nn.functional.interpolate(pseudo_label[None, None, :],
                                                               size=visual_feat_length,
                                                               mode='nearest').flatten()  # l -> L
                pseudo_label = truncate_or_pad_feature_length(pseudo_label, target_length=self.vis_max_len)[0]

        # Prepare the result dictionary
        res_dict = dict(v_feat=visual_feat, v_feat_l=visual_feat_length,
                        t_feat=text_feat, t_feat_l=text_feature_length,
                        label=label, target=target, text=text_query,
                        video_path=vid_path, pseudo_label=pseudo_label)

        return res_dict

    def __len__(self):
        return len(self.gts)


class PreVADSynthesisDataset(Dataset):
    def __init__(self,
                 visual_feature_dir,
                 gt_path,
                 vis_max_len: int = 128,
                 txt_max_len: int = 77,
                 ) -> None:
        self.visual_feat_dir = Path(visual_feature_dir)
        # self.text_feature_dir = Path(text_feature_dir)
        with open(gt_path) as f:
            self.gts = json.load(f)  # ground truths

        self.vis_max_len = vis_max_len
        self.txt_max_len = txt_max_len

    def __getitem__(self, item):
        # Load visual features
        vid_path = self.gts[item]['path']
        vid = vid_path.replace('.npy', '')
        visual_feat = torch.tensor(np.load(self.visual_feat_dir / vid_path), dtype=torch.float32)
        visual_feat, visual_feat_length = truncate_or_pad_feature_length(
            visual_feat, target_length=self.vis_max_len
        )
        visual_feat_length = torch.tensor(visual_feat_length)

        # Load Text
        text_query = self.gts[item]['anomaly_text']
        if type(text_query) is list:
            text_query = random.choice(text_query)

        # get temporal pseudo annotation
        norm_span = torch.tensor(self.gts[item]['anomaly_span'], dtype=torch.float)
        norm_span = (norm_span * visual_feat_length).long()
        pseudo_label = torch.zeros(self.vis_max_len, dtype=torch.float)
        pseudo_label[norm_span[0]:norm_span[1]] = 1.0

        # Prepare the result dictionary
        res_dict = dict(v_feat=visual_feat, v_feat_l=visual_feat_length,
                        text=text_query,
                        video_path=vid_path,
                        pseudo_label=pseudo_label, pseudo_span=norm_span,
                        temp_anno=norm_span[None, :])

        return res_dict

    def __len__(self):
        return len(self.gts)


class PreVADSynthesisOVDataset(Dataset):
    def __init__(self,
                 visual_feature_dir,
                 gt_path,
                 label_path,
                 vis_max_len: int = 512,
                 ) -> None:
        self.visual_feat_dir = Path(visual_feature_dir)
        with open(gt_path) as f:
            self.gts = json.load(f)  # ground truths
        with open(label_path) as f:
            label_info = json.load(f)
            self.labels: list = label_info['labels']
            self.label2text: dict[list] = label_info['label2text']

        self.vis_max_len = vis_max_len

    def __getitem__(self, item):
        # Load visual features
        vid_path = self.gts[item]['path']
        vid = vid_path.replace('.npy', '')
        visual_feat = torch.tensor(np.load(self.visual_feat_dir / vid_path), dtype=torch.float32)
        visual_feat, visual_feat_length = truncate_or_pad_feature_length(
            visual_feat, target_length=self.vis_max_len
        )
        visual_feat_length = torch.tensor(visual_feat_length)

        # Load Text
        label_cls = self.gts[item]['anomaly_cls']
        index_cls = torch.tensor(self.labels.index(label_cls))
        rand_label_text = [random.choice(self.label2text[l]) for l in self.labels]
        text_cls = rand_label_text[index_cls]

        # get temporal pseudo annotation
        norm_span = torch.tensor(self.gts[item]['anomaly_span'], dtype=torch.float)
        norm_span = (norm_span * visual_feat_length).long()
        pseudo_frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
        pseudo_frame_label[norm_span[0]:norm_span[1]] = 1.0

        # Prepare the result dictionary
        res_dict = dict(
            v_feat=visual_feat, v_feat_l=visual_feat_length, target_length=visual_feat_length,
            label=index_cls, labels=rand_label_text,
            video_path=vid_path,
            pseudo_frame_label=pseudo_frame_label, pseudo_span=norm_span,
            temp_anno=norm_span[None, :]  # only for evaluation
        )

        return res_dict

    def __len__(self):
        return len(self.gts)


class PreVADDataset(Dataset):
    def __init__(self,
                 data_root,
                 gt_json_path,
                 vis_max_len: int = 128,
                 txt_max_len: int = 77,
                 class_names: Optional[List[str]] = None
                 ):
        self.abbr = 'prevad'
        self.root = Path(data_root)
        with open(gt_json_path) as f:
            self.gts = json.load(f)

        self.vis_max_len = vis_max_len
        self.txt_max_len = txt_max_len
        self.max_span_num = 6

        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = list(set([i['class_name'] for i in self.gts]))
            self.class_names.sort()

        self.num_overlen_clips = 0

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
            normed_span = (normed_span * visual_feat_length).long()
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


class PreVADDatasetOnline(Dataset):
    def __init__(self,
                 data_root,
                 gt_json_path,
                 max_num_clips: int = 5,
                 vis_max_len: int = 256,
                 txt_max_len: int = 77,
                 class_names: Optional[List[str]] = None,
                 heuristic_synthesis: bool = False,
                 enhance_single_clip_factor: float = 0.0,
                 random_crop_normal: bool = False,
                 retrieval_based_synthesis: bool = False,
                 retrieval_cache: str = None,
                 ):
        self.abbr = 'prevad'
        self.root = Path(data_root)
        with open(gt_json_path) as f:
            self.gts = json.load(f)

        self.anomaly_gts = [i for i in self.gts if i['class_name'] != 'Normal']
        self.normal_gts = [i for i in self.gts if i['class_name'] == 'Normal']
        self.num_anomaly = len(self.anomaly_gts)
        self.num_normal = len(self.anomaly_gts)  # normal synthesis == abnormal synthesis
        self.vid2gt = {Path(i['video_path']).stem: i for i in self.gts}

        self.vis_max_len = vis_max_len
        self.txt_max_len = txt_max_len
        self.max_num_clips = max_num_clips
        assert 0.0 <= enhance_single_clip_factor <= 1.0
        self.enhance_single_clip_factor = enhance_single_clip_factor
        self.random_crop_normal = random_crop_normal
        self.max_span_num = 6

        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = list(set([i['class_name'] for i in self.gts]))
            self.class_names.sort()

        if retrieval_based_synthesis is True:
            with open(retrieval_cache) as f:
                self.ret_cache = json.load(f)

        self.norm_src_to_gts = None
        self.heuristic_synthesis = heuristic_synthesis
        self.retrieval_based_synthesis = retrieval_based_synthesis
        self.num_overlen_clips = 0

    def _build_norm_src_to_gts(self):
        """
        norm srcs:
        RWF-2000 VIDAL caption_datasets
        road_camera fp_driving web_camera animal_camera city_walk
        factory
        """
        if self.norm_src_to_gts is None:
            print('Building norm_src_to_gts ...')
            self.norm_src_to_gts = defaultdict(list)
            for gt in self.normal_gts:
                self.norm_src_to_gts[gt['norm_src']].append(gt)
            print(f'Finish building norm_src_to_gts: {list(self.norm_src_to_gts.keys())}')

    def _retrieve_normal_clip(self, query_vid=None, query_type=None):
        """
        Args:
            query_vid:
            query_type: one of anomaly classes

        Returns:

        """
        if query_vid is None:
            if query_type is None:
                normal_item = random.choice(self.normal_gts)
                normal_path = normal_item['path'].replace('\n', '')  # some bad names contain \n
                feat = torch.tensor(np.load(self.root / normal_path), dtype=torch.float32)  # [T, C]
                if self.random_crop_normal is True:
                    crop_start = random.randint(0, len(feat) // 2)
                    crop_end = random.randint((crop_start + len(feat)) // 2, len(feat))
                    feat = feat[crop_start:crop_end]
                return feat, normal_path
            else:
                self._build_norm_src_to_gts()
                if query_type in ['RWF-2000', 'VIDAL', 'caption_datasets', 'road_camera', 'fp_driving', 'web_camera', 'animal_camera', 'city_walk', 'factory']:
                    normal_item = random.choice(self.norm_src_to_gts[query_type])
                elif query_type in ['MechanicalAccident']:
                    rand_num = random.random()
                    if rand_num < 0.7:
                        normal_item = random.choice(self.norm_src_to_gts['factory'])
                    elif rand_num < 0.9:
                        normal_item = random.choice(self.norm_src_to_gts['web_camera'])
                    else:
                        normal_item = random.choice(self.normal_gts)
                elif query_type in ['CarAccident', 'TrainAccident', 'AirAccident']:
                    rand_num = random.random()
                    if rand_num < 0.5:
                        normal_item = random.choice(self.norm_src_to_gts['fp_driving'])
                    elif rand_num < 0.8:
                        normal_item = random.choice(self.norm_src_to_gts['road_camera'])
                    elif rand_num < 0.9:
                        normal_item = random.choice(self.norm_src_to_gts['web_camera'])
                    else:
                        normal_item = random.choice(self.normal_gts)
                elif query_type in ['AnimalAttackAnimal', 'AnimalAttackHuman', 'AnimalPredation']:
                    rand_num = random.random()
                    if rand_num < 0.1:
                        normal_item = random.choice(self.norm_src_to_gts['animal_camera'])
                    elif rand_num < 0.3:
                        normal_item = random.choice(self.norm_src_to_gts['city_walk'])
                    elif rand_num < 0.5:
                        normal_item = random.choice(self.norm_src_to_gts['web_camera'])
                    else:
                        normal_item = random.choice(self.normal_gts)
                elif query_type in ['Violence', 'Robbery', 'Riot', 'Shooting', 'RangeShooting', 'WarScene', 'CrowdViolence']:
                    rand_num = random.random()
                    if rand_num < 0.1:
                        normal_item = random.choice(self.norm_src_to_gts['RWF-2000'])
                    elif rand_num < 0.7:
                        normal_item = random.choice(self.norm_src_to_gts['web_camera'])
                    elif rand_num < 0.9:
                        normal_item = random.choice(self.norm_src_to_gts['city_walk'])
                    else:
                        normal_item = random.choice(self.normal_gts)
                elif query_type in ['Collapse', 'Fire']:
                    rand_num = random.random()
                    if rand_num < 0.2:
                        normal_item = random.choice(self.norm_src_to_gts['VIDAL'])
                    elif rand_num < 0.4:
                        normal_item = random.choice(self.norm_src_to_gts['caption_datasets'])
                    elif rand_num < 0.9:
                        normal_item = random.choice(self.norm_src_to_gts['web_camera'])
                    else:
                        normal_item = random.choice(self.normal_gts)
                elif query_type in ['FallDown', 'FallIntoWater', 'Accident', 'ObjectImpact', 'Explosion']:
                    rand_num = random.random()
                    if rand_num < 0.1:
                        normal_item = random.choice(self.norm_src_to_gts['road_camera'])
                    elif rand_num < 0.2:
                        normal_item = random.choice(self.norm_src_to_gts['fp_driving'])
                    elif rand_num < 0.3:
                        normal_item = random.choice(self.norm_src_to_gts['city_walk'])
                    elif rand_num < 0.9:
                        normal_item = random.choice(self.norm_src_to_gts['web_camera'])
                    else:
                        normal_item = random.choice(self.normal_gts)
                else:
                    normal_item = random.choice(self.normal_gts)
                normal_path = normal_item['path'].replace('\n', '')  # some bad names contain \n
                feat = torch.tensor(np.load(self.root / normal_path), dtype=torch.float32)  # [T, C]
                return feat, normal_path
        else:
            assert self.retrieval_based_synthesis is True and hasattr(self, 'ret_cache')
            if random.random() < 0.5:
                return self._retrieve_normal_clip()
            else:
                normal_vid = random.choice(self.ret_cache[query_vid])
                normal_path = self.vid2gt[normal_vid]['path']
                feat = torch.tensor(np.load(self.root / normal_path), dtype=torch.float32)  # [T, C]
                if self.random_crop_normal is True:
                    crop_start = random.randint(0, len(feat) // 2)
                    crop_end = random.randint((crop_start + len(feat)) // 2, len(feat))
                    feat = feat[crop_start:crop_end]
                return feat, normal_path



    def __getitem__(self, item):
        # if anomaly
        if item < self.num_anomaly:
            gt = self.anomaly_gts[item]

            # Load visual features
            visual_feat = torch.tensor(np.load(self.root / gt['path']), dtype=torch.float32)
            if self.enhance_single_clip_factor > 0.0 and random.random() < self.enhance_single_clip_factor:
                num_clips = 1
            else:
                num_clips = random.randint(1, self.max_num_clips)  # including the anomaly clip

            clips, clip_paths = [], []
            insert_idx = random.randint(0, num_clips - 1)
            clips_len = []
            for i in range(num_clips):
                if i == insert_idx:
                    clips.append(visual_feat)
                    clip_paths.append(gt['path'])
                else:
                    if self.heuristic_synthesis:
                        norm_feat, norm_path = self._retrieve_normal_clip(query_type=gt['class_name'])
                        clips.append(norm_feat)
                        clip_paths.append(norm_path)
                    elif self.retrieval_based_synthesis:
                        norm_feat, norm_path = self._retrieve_normal_clip(query_vid=Path(gt['path']).stem)
                        clips.append(norm_feat)
                        clip_paths.append(norm_path)
                    else:
                        norm_feat, norm_path = self._retrieve_normal_clip()
                        clips.append(norm_feat)
                        clip_paths.append(norm_path)
                clips_len.append(len(clips[-1]))
            clips_len = np.cumsum(np.array([0, ] + clips_len))
            clips_range = np.stack([clips_len[:-1], clips_len[1:]], axis=1)  # T,2
            clips_feat = torch.cat(clips, dim=0)  # T,D
            clips_range = torch.tensor(clips_range)
            if len(clips_feat) > self.vis_max_len:  # for debug
                self.num_overlen_clips += 1

            visual_feat, visual_feat_length = truncate_or_pad_feature_length(
                clips_feat, target_length=self.vis_max_len
            )
            visual_feat_length = torch.tensor(visual_feat_length)


            # Load CLS
            class_name = gt['class_name']
            class_idx = torch.tensor(self.class_names.index(class_name))

            # Load Description
            descs = gt.get('descriptions', None)
            if type(descs) is list:
                desc = random.choice(descs)
            elif type(descs) is str:
                desc = descs
            else:
                desc = ""

            frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
            ano_span = clips_range[insert_idx]
            frame_label[ano_span[0]:ano_span[1]] = 1.0
            span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
            span_ts *= -1
            span_ts[0, :] = ano_span

        # if normal
        else:
            if self.enhance_single_clip_factor > 0.0 and random.random() < self.enhance_single_clip_factor:
                num_clips = 1
            else:
                num_clips = random.randint(1, self.max_num_clips)  # including the anomaly clip
            prior_norm_gt = random.choice(self.normal_gts)
            prior_norm_path = prior_norm_gt['path'].replace('\n', '')  # some bad names contain \n
            prior_norm_feat = torch.tensor(np.load(self.root / prior_norm_path), dtype=torch.float32)  # [T, C]
            clips, clip_paths = [prior_norm_feat], [prior_norm_path]
            clips_len = [len(prior_norm_feat)]

            for i in range(num_clips-1):
                if self.heuristic_synthesis:
                    norm_feat, norm_path = self._retrieve_normal_clip(query_type=prior_norm_gt['norm_src'])
                    clips.append(norm_feat)
                    clip_paths.append(norm_path)
                elif self.retrieval_based_synthesis:
                    norm_feat, norm_path = self._retrieve_normal_clip(query_vid=Path(prior_norm_gt['path']).stem)
                    clips.append(norm_feat)
                    clip_paths.append(norm_path)
                else:
                    norm_feat, norm_path = self._retrieve_normal_clip()
                    clips.append(norm_feat)
                    clip_paths.append(norm_path)
                clips_len.append(len(clips[-1]))
            clips_len = np.cumsum(np.array([0, ] + clips_len))
            clips_range = np.stack([clips_len[:-1], clips_len[1:]], axis=1)  # T,2
            clips_feat = torch.cat(clips, dim=0)  # T,D
            if len(clips_feat) > self.vis_max_len:  # for debug
                self.num_overlen_clips += 1

            visual_feat, visual_feat_length = truncate_or_pad_feature_length(
                clips_feat, target_length=self.vis_max_len
            )
            visual_feat_length = torch.tensor(visual_feat_length)

            # Load CLS Text & Description
            class_name = 'Normal'
            class_idx = torch.tensor(0)
            desc = ""

            frame_label = torch.zeros(self.vis_max_len, dtype=torch.float)
            span_ts = torch.ones(self.max_span_num, 2, dtype=torch.long)
            span_ts *= -1

        # Prepare the result dictionary
        res_dict = dict(
            v_feat=visual_feat, v_feat_l=visual_feat_length, target_length=visual_feat_length,
            cls_label=class_name, cls_label_idx=class_idx,
            desc_label=desc,
            pseudo_frame_label=frame_label,
            pseudo_span=span_ts,
            temp_span=span_ts,  # only for evaluation
            _clip_paths="\n".join(clip_paths),  # only for visualization
            _clips_range=str(clips_range.tolist()),  # only for visualization
        )

        return res_dict

    def __len__(self):
        return self.num_anomaly * 2


class PreVADSampleFeatureDataset(PreVADFeatureDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 30


class PreVADSampleSynthesisDataset(PreVADSynthesisDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 30


class PreVADSampleSynthesisOVDataset(PreVADSynthesisOVDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 30


# class PreVADDataModule(BaseDataModule):
#
#     def __init__(self,
#                  train_data_root: str,
#                  val_data_root: str,
#                  batch_size: int = 16,
#                  heuristic_synthesis=False,
#                  syn_max_num_clips=5,
#                  vis_max_len=256,
#                  ) -> None:
#         super().__init__()
#         self.train_data_root = Path(train_data_root)
#         self.val_data_root = Path(val_data_root)
#         self.batch_size = batch_size
#
#         self.test_sets = []
#         self.val_sets = []
#         self.train_set = None
#
#         self.heuristic_synthesis = heuristic_synthesis
#         self.max_num_clips = syn_max_num_clips
#         self.vis_max_len = vis_max_len
#
#     def setup(self, stage: str) -> None:
#         if stage == 'fit':
#             # self.train_set = PreVADDataset(
#             #     self.train_data_root,
#             #     self.train_data_root / 'v2' / 'train_anno.json',
#             #     class_names=DEFAULT_CLASSES,
#             #     vis_max_len=256,
#             # )
#             self.train_set = PreVADDatasetOnline(
#                 self.train_data_root,
#                 self.train_data_root / 'v3' / 'train_anno.json',
#                 class_names=DEFAULT_CLASSES,
#                 vis_max_len=self.vis_max_len,
#                 heuristic_synthesis=self.heuristic_synthesis,
#                 max_num_clips=self.max_num_clips,
#             )
#         self.val_sets.append(
#             PreVADDataset(
#                 self.train_data_root,
#                 self.train_data_root / 'v2' / 'test_anno.json',
#                 class_names=DEFAULT_CLASSES,
#                 vis_max_len=self.vis_max_len,
#             )
#         )
#         self.val_sets.append(
#             XDDataset(
#                 self.train_data_root,
#                 self.train_data_root / 'other_datasets' / 'xd_test_anno.json',
#             )
#         )
#         self.val_sets.append(
#             UCFCrimeDataset(
#                 self.train_data_root,
#                 self.train_data_root / 'other_datasets' / 'ucf_test_anno.json',
#             )
#         )
#
#     def train_dataloader(self):
#         return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
#                           num_workers=4, pin_memory=True)
#
#     def val_dataloader(self):
#         loaders = []
#         for val_set in self.val_sets:
#             loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
#                                       num_workers=4, pin_memory=True))
#         return loaders
#
#     def test_dataloader(self):
#         loaders = []
#         for val_set in self.val_sets:
#             loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
#                                       num_workers=4, pin_memory=True))
#         return loaders
