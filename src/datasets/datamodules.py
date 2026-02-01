from pathlib import Path

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from .base import BaseDataModule
from .PreVAD import PreVADDatasetOnline, PreVADDataset
from .PreVAD import DEFAULT_CLASSES as PREVAD_DEFAULT_CLASSES
from .UCFCrime import UCFCrimeDataset
from .XDViolence import XDDataset
from .MSAD import MSADDataset
from .DoTA import DoTADataset
from .UBNormal import UBNormalDataset
from .TAD import TADDataset
from .LAD import LADDataset


class PreVADDataModule(BaseDataModule):

    def __init__(self,
                 train_data_root: str,
                 val_data_root: str,
                 batch_size: int = 16,
                 heuristic_synthesis=False,
                 retrieval_based_synthesis: bool = False,
                 retrieval_cache: str = None,
                 enhance_single_clip_factor: float = 0.0,
                 random_crop_normal: bool = False,
                 syn_max_num_clips=5,
                 vis_max_len=256,
                 eval_datasets: str = "prevad,xd,ucf",
                 ) -> None:
        super().__init__()
        self.train_data_root = Path(train_data_root)
        self.val_data_root = Path(val_data_root)
        self.batch_size = batch_size

        self.val_sets = []
        self.train_set = None

        self.heuristic_synthesis = heuristic_synthesis
        self.enhance_single_clip_factor = enhance_single_clip_factor
        self.random_crop_normal = random_crop_normal
        self.retrieval_based_synthesis = retrieval_based_synthesis
        self.retrieval_cache = str(self.train_data_root / retrieval_cache)
        self.max_num_clips = syn_max_num_clips
        self.vis_max_len = vis_max_len
        self.eval_dataset_names = eval_datasets.split(',')

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set = PreVADDatasetOnline(
                self.train_data_root,
                self.train_data_root / 'v6' / 'prevad_train_anno_v6.json',
                class_names=PREVAD_DEFAULT_CLASSES,
                vis_max_len=self.vis_max_len,
                heuristic_synthesis=self.heuristic_synthesis,
                enhance_single_clip_factor=self.enhance_single_clip_factor,
                random_crop_normal=self.random_crop_normal,
                max_num_clips=self.max_num_clips,
                retrieval_based_synthesis=self.retrieval_based_synthesis,
                retrieval_cache=self.retrieval_cache,
            )
        if 'prevad' in self.eval_dataset_names:
            self.val_sets.append(
                PreVADDataset(
                    self.train_data_root,
                    self.train_data_root / 'v6' / 'prevad_test_anno_v6.json',
                    class_names=PREVAD_DEFAULT_CLASSES,
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'xd' in self.eval_dataset_names:
            self.val_sets.append(
                XDDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'xd_test_anno.json',
                    vis_max_len=512,
                )
            )
        if 'ucf' in self.eval_dataset_names:
            self.val_sets.append(
                UCFCrimeDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'ucf_test_anno.json',
                    vis_max_len=512,
                )
            )
        if 'msad' in self.eval_dataset_names:
            self.val_sets.append(
                MSADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'msad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'ubnormal' in self.eval_dataset_names:
            self.val_sets.append(
                UBNormalDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'ubnormal_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'dota' in self.eval_dataset_names:
            self.val_sets.append(
                DoTADataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'dota_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'tad' in self.eval_dataset_names:
            self.val_sets.append(
                TADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'tad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'lad' in self.eval_dataset_names:
            self.val_sets.append(
                LADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'lad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=32, pin_memory=True)

    def val_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders

    def test_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders

    def predict_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders


class UCFFinetuneDataModule(BaseDataModule):
    def __init__(self,
                 data_root: str,
                 batch_size: int = 16,
                 vis_max_len=256,
                 ten_crop_train=False,
                 ten_crop_test=False,
                 ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.vis_max_len = vis_max_len
        self.ten_crop_train = ten_crop_train
        self.ten_crop_test = ten_crop_test
        self.train_set, self.val_set = None, None

    def setup(self, stage: str) -> None:
        if self.ten_crop_train:
            train_json_path = self.data_root / 'other_datasets' / 'ucf_train_10crop_anno.json'
        else:
            train_json_path = self.data_root / 'other_datasets' / 'ucf_train_anno.json'
        if self.ten_crop_test:
            test_json_path = self.data_root / 'other_datasets' / 'ucf_test_10crop_anno.json'
        else:
            test_json_path = self.data_root / 'other_datasets' / 'ucf_test_anno.json'

        if stage == 'fit':
            self.train_set = UCFCrimeDataset(
                self.data_root,
                train_json_path,
                vis_max_len=self.vis_max_len,
            )
            self.val_set = UCFCrimeDataset(
                self.data_root,
                test_json_path,
                vis_max_len=self.vis_max_len,
            )
        else:
            self.val_set = UCFCrimeDataset(
                self.data_root,
                test_json_path,
                vis_max_len=self.vis_max_len,
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)


class UCFFinetuneCrossDataModule(BaseDataModule):

    def __init__(self,
                 train_data_root: str,
                 val_data_root: str,
                 batch_size: int = 16,
                 vis_max_len=256,
                 eval_datasets: str = "xd,ucf",
                 ) -> None:
        super().__init__()
        self.train_data_root = Path(train_data_root)
        self.val_data_root = Path(val_data_root)
        self.batch_size = batch_size

        self.val_sets = []
        self.train_set = None

        self.vis_max_len = vis_max_len
        self.eval_dataset_names = eval_datasets.split(',')

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set = UCFCrimeDataset(
                self.train_data_root,
                self.train_data_root / 'other_datasets' / 'ucf_train_10crop_anno.json',
                vis_max_len=self.vis_max_len,
            )
        if 'xd' in self.eval_dataset_names:
            self.val_sets.append(
                XDDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'xd_test_anno.json',
                    vis_max_len=512,
                )
            )
        if 'ucf' in self.eval_dataset_names:
            self.val_sets.append(
                UCFCrimeDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'ucf_test_anno.json',
                    vis_max_len=512,
                )
            )
        if 'msad' in self.eval_dataset_names:
            self.val_sets.append(
                MSADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'msad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'ubnormal' in self.eval_dataset_names:
            self.val_sets.append(
                UBNormalDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'ubnormal_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'dota' in self.eval_dataset_names:
            self.val_sets.append(
                DoTADataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'dota_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'tad' in self.eval_dataset_names:
            self.val_sets.append(
                TADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'tad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'lad' in self.eval_dataset_names:
            self.val_sets.append(
                LADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'lad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=32, pin_memory=True)

    def val_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders

    def test_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders

    def predict_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders


class XDFinetuneDataModule(BaseDataModule):
    def __init__(self,
                 data_root: str,
                 batch_size: int = 16,
                 vis_max_len=256,
                 ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.vis_max_len = vis_max_len
        self.train_set, self.val_set = None, None

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set = XDDataset(
                self.data_root,
                self.data_root / 'other_datasets' / 'xd_train_anno.json',
                vis_max_len=self.vis_max_len,
            )
            self.val_set = XDDataset(
                self.data_root,
                self.data_root / 'other_datasets' / 'xd_test_anno.json',
                vis_max_len=self.vis_max_len,
            )
        else:
            self.val_set = XDDataset(
                self.data_root,
                self.data_root / 'other_datasets' / 'xd_test_anno.json',
                vis_max_len=self.vis_max_len,
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)


class XDFinetuneCrossDataModule(BaseDataModule):

    def __init__(self,
                 train_data_root: str,
                 val_data_root: str,
                 batch_size: int = 16,
                 vis_max_len=256,
                 eval_datasets: str = "xd,ucf",
                 ) -> None:
        super().__init__()
        self.train_data_root = Path(train_data_root)
        self.val_data_root = Path(val_data_root)
        self.batch_size = batch_size

        self.val_sets = []
        self.train_set = None

        self.vis_max_len = vis_max_len
        self.eval_dataset_names = eval_datasets.split(',')

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set = XDDataset(
                self.train_data_root,
                self.train_data_root / 'other_datasets' / 'xd_train_anno.json',
                vis_max_len=self.vis_max_len,
            )
        if 'xd' in self.eval_dataset_names:
            self.val_sets.append(
                XDDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'xd_test_anno.json',
                    vis_max_len=512,
                )
            )
        if 'ucf' in self.eval_dataset_names:
            self.val_sets.append(
                UCFCrimeDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'ucf_test_anno.json',
                    vis_max_len=512,
                )
            )
        if 'msad' in self.eval_dataset_names:
            self.val_sets.append(
                MSADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'msad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'ubnormal' in self.eval_dataset_names:
            self.val_sets.append(
                UBNormalDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'ubnormal_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'dota' in self.eval_dataset_names:
            self.val_sets.append(
                DoTADataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'dota_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'tad' in self.eval_dataset_names:
            self.val_sets.append(
                TADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'tad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )
        if 'lad' in self.eval_dataset_names:
            self.val_sets.append(
                LADDataset(
                    self.train_data_root,
                    self.train_data_root / 'other_datasets' / 'lad_test_anno.json',
                    vis_max_len=self.vis_max_len,
                )
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=32, pin_memory=True)

    def val_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders

    def test_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders

    def predict_dataloader(self):
        loaders = []
        for val_set in self.val_sets:
            loaders.append(DataLoader(val_set, batch_size=64, shuffle=False,
                                      num_workers=8, pin_memory=True))
        return loaders
