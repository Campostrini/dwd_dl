from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import LightningModule

from dwd_dl import cfg


class LoggerCallbkack(Callback):
    def __init__(self, *args, **kwargs):
        self._something = None

    def on_train_start(self, trainer, pl_module: LightningModule) -> None:
        # pl_module.logger.log_hyperparams(pl_module.hparams)
        pl_module.logger.experiment.add_hparams(
            dict(pl_module.hparams),
            {"hp/valid_loss": 2, "hp/valid_accuracy": 0}
        )

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: Any) -> None:
        # TODO: implement when available
        # pl_module.logger.experiment.add_hparams(
        #     dict(pl_module.hparams),
        #     {
        #         'hparam/train_loss': train_loss,
        #         'hparam/train_acc': train_acc,
        #     },
        # )
        pass

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        # TODO: move from model to here when available.
        pass


class CallbacksList:
    def __init__(self, experiment_timestamp_str):
        val_loss_checkpoint = ModelCheckpoint(
            monitor='val/loss',
            dirpath=cfg.CFG.create_checkpoint_dir(),
            filename=f'{experiment_timestamp_str}' + '-{epoch:02d}-{valid_loss:.2f}',
            save_top_k=2,
            mode='min',
        )

        logger_callback = LoggerCallbkack()

        learning_rate_monitor = LearningRateMonitor(logging_interval='step')

        self.callback_list = [val_loss_checkpoint, logger_callback, learning_rate_monitor]

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

