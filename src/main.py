from lightning import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from models import VADCBase
from datasets import BaseDataModule
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from utils import SavePredResults

torch.set_float32_matmul_precision('high')


# hack and fix lightning progress bar in PyCharm
class FixTQDMProgressBar(TQDMProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        bar.leave = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        bar.leave = True
        return bar


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, WandbLogger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility

            save_dir = trainer.logger.save_dir
            name = trainer.logger.name
            version = trainer.logger.version
            yaml_path = Path(save_dir, str(name), version, "config.yaml")
            yaml_path.parent.mkdir(exist_ok=False, parents=True)
            with open(yaml_path, "w+", encoding='utf-8') as f:
                f.write(config)


def main():
    cli = LightningCLI(
        subclass_mode_model=True, model_class=VADCBase,
        subclass_mode_data=True, datamodule_class=BaseDataModule,
        parser_kwargs={
            'fit': {'default_config_files': ["configs/default.yaml"]},
            'validate': {'default_config_files': ["configs/default.yaml"]},
            'test': {'default_config_files': ["configs/default.yaml"]},
        },
        save_config_kwargs={"overwrite": True},
        save_config_callback=LoggerSaveConfigCallback,
        # run=False
    )
    # cli.trainer.fit(cli.model)
    # if cli.subcommand == 'fit':
    #     cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path='best')


if __name__ == '__main__':
    main()
