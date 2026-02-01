import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Union, Tuple
import random


def resample_or_pad_feature_length(input_feature: Union[torch.Tensor, np.ndarray],
                                   target_length: int,
                                   resample_mode: str = 'uniform') -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    """
    Resamples or pads the length of an input feature to a target length.

    Args:
        input_feature (Union[torch.Tensor, np.ndarray]): The input feature to be resampled or padded.
            Its shape should be [L,*], where * represents any number of dimensions.
        target_length (int): The desired length of the output feature.
        resample_mode (str, optional): The resampling mode. Can be either 'uniform' or 'random'. Defaults to 'uniform'.

    Returns:
        Tuple[Union[torch.Tensor, np.ndarray], int]: A tuple containing the output feature and its length.
    """
    l_feat = len(input_feature)
    is_tensor = isinstance(input_feature, torch.Tensor)
    if l_feat > target_length:
        if resample_mode == 'uniform':
            r = np.linspace(0, l_feat - 1, target_length, dtype=np.int16)
            output_feature = input_feature[torch.tensor(r, dtype=torch.int)] if is_tensor else input_feature[r]
        elif resample_mode == 'random':
            r = np.random.randint(l_feat - target_length)
            r = torch.tensor(r) if is_tensor else r
            output_feature = input_feature[r:r + target_length]
        else:
            raise ValueError
        output_length = target_length
    elif l_feat < target_length:
        pad_l = target_length - l_feat
        ext_shape = input_feature.shape[1:]
        if is_tensor:
            output_feature = torch.cat([
                input_feature,
                torch.zeros((pad_l,) + ext_shape, device=input_feature.device, dtype=input_feature.dtype)
            ], dim=0)
        else:
            output_feature = np.pad(input_feature, (0, pad_l), mode='constant', constant_values=0)
        output_length = l_feat
    else:
        output_feature = input_feature
        output_length = l_feat
    return output_feature, output_length


def truncate_or_pad_feature_length(input_feature: Union[torch.Tensor, np.ndarray],
                                   target_length: int) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    l_feat = len(input_feature)
    is_tensor = isinstance(input_feature, torch.Tensor)
    if l_feat >= target_length:
        output_feature = input_feature[:target_length]
        output_length = target_length
    else:
        pad_l = target_length - l_feat
        ext_shape = input_feature.shape[1:]
        if is_tensor:
            output_feature = torch.cat([
                input_feature,
                torch.zeros((pad_l,) + ext_shape, device=input_feature.device, dtype=input_feature.dtype)
            ], dim=0)
        else:
            output_feature = np.pad(input_feature, (0, pad_l), mode='constant', constant_values=0)
        output_length = l_feat
    return output_feature, output_length


class BinaryBalancedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, cls_ratio=0.5, shuffle=False):
        super().__init__()
        if type(data_source.labels) is not torch.Tensor:
            labels = torch.tensor(data_source.labels)
        else:
            labels = data_source.labels
        self.normal_indices = torch.where(labels == 0)[0].tolist()
        self.abnormal_indices = torch.where(labels == 1)[0].tolist()

        self.cls_ratio = cls_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_nor = round(self.cls_ratio * self.batch_size)
        self.num_abn = self.batch_size - self.num_nor

    @property
    def _len_normal_batch(self):
        return (len(self.normal_indices) + self.num_nor - 1) // self.num_nor

    @property
    def _len_abnormal_batch(self):
        return (len(self.abnormal_indices) + self.num_abn - 1) // self.num_abn

    def __iter__(self):
        if self.shuffle is True:
            random.shuffle(self.normal_indices)
            random.shuffle(self.abnormal_indices)

        batch = []
        nor_idx, abn_idx = 0, 0
        reach_end = [False, False]
        while True:
            if reach_end[0] is True and reach_end[1] is True:
                return

            # Add normal to the batch
            # if the first time reach end, reset the index to 0 and go
            # if the second time reach end, stop
            for _ in range(self.num_nor):
                if nor_idx >= len(self.normal_indices):
                    reach_end[0] = True
                    nor_idx = 0
                    if reach_end[0] is True and reach_end[1] is True:
                        break
                batch.append(self.normal_indices[nor_idx])
                nor_idx += 1
            # add abnormal to the batch
            # if the first time reach end, reset the index to 0 and go
            # if the second time reach end, stop
            for _ in range(self.num_abn):
                if abn_idx >= len(self.abnormal_indices):
                    reach_end[1] = True
                    abn_idx = 0
                    if reach_end[0] is True and reach_end[1] is True:
                        break
                batch.append(self.abnormal_indices[abn_idx])
                abn_idx += 1

            yield batch
            batch = []


    def __len__(self):
        return max(self._len_abnormal_batch, self._len_normal_batch)


if __name__ == '__main__':
    class TMP_DS:
        # labels = (torch.randn(400) > 0.4).long()
        labels = torch.cat([torch.ones(200), torch.zeros(200)])
    sampler = BinaryBalancedBatchSampler(TMP_DS(), 16, 0.4, True)
    # print(TMP_DS.labels)
    print(sampler._len_normal_batch, sampler._len_abnormal_batch, len(sampler))
    print(len(sampler.normal_indices), len(sampler.abnormal_indices))
    print(sampler.num_nor, sampler.num_abn)
    for i, item in enumerate(sampler):
        print(f"{i+1}: {len(item)}: {item}")
    # iterator = iter(sampler)
    # print(next(iterator))
    # print(next(iterator))
    # print(next(iterator))
