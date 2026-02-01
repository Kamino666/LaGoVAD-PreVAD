import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

import pickle
from pathlib import Path
from collections import defaultdict


class SavePredResults(Callback):
    def __init__(self, output_dir: str):
        if output_dir is None or output_dir == '':
            self.output_dir = None
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not hasattr(pl_module, 'pred_results'):
            print('no pred_results in pl_module, SavePredResults callback not working.')
            return
        if self.output_dir is None:  # pass
            return

        # pred_results_by_dataset
        #   0: {video_path: str, score: numpy, dataloader_idx: int}
        #   1: {video_path: str, score: numpy, dataloader_idx: int}
        #   ...
        pred_results_by_dataset = defaultdict(list)
        for pred_res in pl_module.pred_results:
            pred_results_by_dataset[pred_res['dataloader_idx']].append(pred_res)
        # save
        pred_path = self.output_dir / 'pred_results.pkl'
        with open(str(pred_path), 'wb+') as f:
            pickle.dump(pred_results_by_dataset, f)
