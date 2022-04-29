from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, EarlyStopping
from pytorch_lightning import LightningModule

from dwd_dl import cfg


class CallbacksList:
    def __init__(self, experiment_timestamp_str):
        val_loss_checkpoint = ModelCheckpoint(
            monitor='val/loss',
            dirpath=cfg.CFG.create_checkpoint_dir(),
            filename=f'{experiment_timestamp_str}' + '-{epoch:02d}-{valid_loss:.2f}',
            save_top_k=2,
            mode='min',
            save_last=True,
        )

        learning_rate_monitor = LearningRateMonitor(logging_interval='step')

        early_stopping = EarlyStopping(monitor="val/epoch_loss", patience=3)

        self.callback_list = [val_loss_checkpoint, learning_rate_monitor, early_stopping]

    def __list__(self):
        return self.callback_list

    def __type__(self):
        return type(self.callback_list)

    def __iter__(self):
        return iter(self.callback_list)

    def __len__(self):
        return len(self.callback_list)

    def __repr__(self):
        return repr(self.callback_list)

    def __add__(self, other):
        return other + self.callback_list

    def append(self, item):
        assert isinstance(item, Callback)
        self.callback_list.append(item)

