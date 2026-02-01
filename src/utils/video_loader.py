import torch
import math
import torchvision.tv_tensors
import numpy as np
from einops.layers.torch import Rearrange
import decord
import ffmpeg
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms.v2 import (
    Normalize, Resize, CenterCrop, ToDtype,
    RandomHorizontalFlip, RandomResizedCrop,
    ToPureTensor, TenCrop
)

import logging
from typing import Union, Optional

decord.bridge.set_bridge('torch')
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class VideoLoader:
    def __init__(self,
                 path,
                 output_size: int = 224,
                 batch_size=8,
                 interval=20,
                 augmentation: Optional[str] = None,
                 rand_temporal_sampling=False,
                 num_aug=5,
                 norm='openai-clip'
                 ):
        self.path = path
        self.interval = interval
        self.bs = batch_size
        self.output_size = output_size
        self.rand_temporal_sampling = rand_temporal_sampling

        if norm == 'openai-clip':
            norm_aug = Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        elif norm == 'imagenet':
            norm_aug = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            norm_aug = torch.nn.Identity()

        self.augmentation = augmentation
        self.num_aug = num_aug
        if augmentation is None:
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                Resize(self.output_size, antialias=True),
                CenterCrop(self.output_size),
                ToPureTensor(),
            ])
        elif augmentation == 'randcrop':
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                RandomResizedCrop(self.output_size, scale=(0.6, 1.0), ratio=(0.9, 1.1), antialias=True),
                RandomHorizontalFlip(),
                ToPureTensor(),
            ])
        elif augmentation == 'tencrop':
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                Resize((340, 256), antialias=True),
                TenCrop(224),
                ToPureTensor(),
            ])
            if rand_temporal_sampling is True:
                raise NotImplementedError
        elif augmentation == 'no_center_crop':
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                Resize((self.output_size, self.output_size), antialias=True),
                ToPureTensor(),
            ])

    @staticmethod
    def _get_num_frames(path):
        streams = ffmpeg.probe(str(path))['streams']
        s = [stream for stream in streams if stream['codec_type'] == 'video'][0]
        duration = float(s['duration'])
        fps = eval(s['avg_frame_rate'])
        num_frames = int(duration * fps)
        return num_frames

    def __iter__(self):
        self._vr = decord.VideoReader(self.path)
        # DO NOT use len(self._vr) to get number of frames, as it may be incorrect!!!
        num_frames = min(self._get_num_frames(self.path), len(self._vr))

        # make read indices
        if self.rand_temporal_sampling:
            sampled_frms = np.arange(0, num_frames, self.interval)[None, :]  # 1, L
            offset = self.interval // self.num_aug
            offsets = np.arange(0, self.num_aug * offset, offset)[:, None]  # num_aug, 1
            sampled_frms = offsets + sampled_frms  # broadcast to num_aug, L
            self._sampled_frms = np.clip(sampled_frms, 0, num_frames - 1)  # make sure valid [num_aug, L]
        else:
            self._sampled_frms = np.arange(self.interval // 2, num_frames, self.interval)[None, :]  # 1, L

        self._curr = 0
        return self

    def __next__(self):
        if self._curr >= self._sampled_frms.shape[1]:
            raise StopIteration
        if self.rand_temporal_sampling is True:
            res = [
                self._vr.get_batch(b[self._curr: self._curr + self.bs])
                for b in self._sampled_frms
            ]
            res = torch.stack(res, dim=0)  # N,T,H,W,C
        else:
            idx = self._sampled_frms[0, self._curr: self._curr + self.bs].tolist()
            res = self._vr.get_batch(idx)
            res = res.unsqueeze(0)  # N,T,H,W,C
        self._curr += self.bs
        res = self.transform(torchvision.tv_tensors.Video(res))
        if type(res) is tuple:  # 10-crop
            res = torch.cat(res, dim=0)  # N,T,C,H,W

        return res  # N,T,C,H,W

    def __len__(self):
        cls = iter(self)
        return math.ceil(self._sampled_frms.shape[1] / self.bs)


"""
This Loader is not used and is much slower for now.
"""
class VideoLoaderTorch:
    def __init__(self,
                 path,
                 output_size: int = 224,
                 batch_size=8,
                 interval=20,
                 augmentation: Optional[str] = None,
                 rand_temporal_sampling=False,
                 num_aug=5,
                 norm='openai-clip'
                 ):
        self.path = path
        self.interval = interval
        self.bs = batch_size
        self.output_size = output_size
        self.rand_temporal_sampling = rand_temporal_sampling

        if norm == 'openai-clip':
            norm_aug = Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        elif norm == 'imagenet':
            norm_aug = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            norm_aug = torch.nn.Identity()

        self.augmentation = augmentation
        self.num_aug = num_aug
        if augmentation is None:
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                Resize(self.output_size, antialias=True),
                CenterCrop(self.output_size),
                ToPureTensor(),
            ])
        elif augmentation == 'randcrop':
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                RandomResizedCrop(self.output_size, scale=(0.6, 1.0), ratio=(0.9, 1.1), antialias=True),
                RandomHorizontalFlip(),
                ToPureTensor(),
            ])
        elif augmentation == 'tencrop':
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                Resize((340, 256), antialias=True),
                TenCrop(224),
                ToPureTensor(),
            ])
            if rand_temporal_sampling is True:
                raise NotImplementedError
        elif augmentation == 'no_center_crop':
            self.transform = Compose([
                Rearrange('b t h w c -> b t c h w'),
                ToDtype(torch.float, scale=True),  # /255
                norm_aug,  # Norm
                Resize((self.output_size, self.output_size), antialias=True),
                ToPureTensor(),
            ])

    def __iter__(self):
        # self._vr = decord.VideoReader(self.path)
        self._reader = torchvision.io.VideoReader(self.path)
        metadata = self._reader.get_metadata()['video']  # pip install av==11.0.0 if something wrong
        num_frames = metadata['fps'][0] * metadata['duration'][0]

        # make read indices
        if self.rand_temporal_sampling:
            sampled_frms = np.arange(0, num_frames, self.interval)[None, :]  # 1, L
            offset = self.interval // self.num_aug
            offsets = np.arange(0, self.num_aug * offset, offset)[:, None]  # num_aug, 1
            sampled_frms = offsets + sampled_frms  # broadcast to num_aug, L
            self._sampled_frms = np.clip(sampled_frms, 0, num_frames - 1)  # make sure valid [num_aug, L]
        else:
            self._sampled_frms = np.arange(self.interval // 2, num_frames, self.interval)[None, :]  # 1, L

        self._curr = 0
        return self

    def _get_batch(self, indices):
        result = []
        for i, frame in enumerate(self._reader):
            if i in indices:
                result.append(frame['data'].permute(1, 2, 0))  # C,H,W -> H,W,C
            if i == indices[-1]:
                break
        return torch.stack(result, dim=0)

    def __next__(self):
        if self._curr >= self._sampled_frms.shape[1]:
            raise StopIteration
        if self.rand_temporal_sampling is True:
            res = [
                self._get_batch(b[self._curr: self._curr + self.bs])
                for b in self._sampled_frms
            ]
            res = torch.stack(res, dim=0)  # N,T,H,W,C
        else:
            idx = self._sampled_frms[0, self._curr: self._curr + self.bs].tolist()
            res = self._get_batch(idx)
            res = res.unsqueeze(0)  # N,T,H,W,C
        self._curr += self.bs
        res = self.transform(torchvision.tv_tensors.Video(res))
        if type(res) is tuple:  # 10-crop
            res = torch.cat(res, dim=0)  # N,T,C,H,W

        return res  # N,T,C,H,W

    def __len__(self):
        cls = iter(self)
        return math.ceil(self._sampled_frms.shape[1] / self.bs)
