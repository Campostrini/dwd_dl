from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from dwd_dl import cfg


class CreateVideo(Callback):
    def __init__(self):
        super().__init__()


class CallbacksList:
    def __init__(self, experiment_timestamp_str):
        val_loss_checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.CFG.create_checkpoint_dir(experiment_timestamp_str),
            filename=f'{experiment_timestamp_str}' + '-epoch{epoch:02d}-val_loss{val_loss:.2f}',
            save_top_k=1,
            mode='min',
            auto_insert_metric_name=False,
        )

        self.callback_list = [val_loss_checkpoint]

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
