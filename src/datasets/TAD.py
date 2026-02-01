from typing import Optional, List
from .base import BaseDataset

DEFAULT_CLASSES = [
    "Normal",
    'Car Accident',
]

class TADDataset(BaseDataset):
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
            max_span_num=15
        )
        self.abbr = 'tad'
