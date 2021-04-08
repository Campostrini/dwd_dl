from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from dwd_dl import cfg


class CallbacksList:
    def __init__(self, experiment_timestamp_str):
        val_loss_checkpoint = ModelCheckpoint(
            monitor='valid_loss',
            dirpath=cfg.CFG.create_checkpoint_dir(experiment_timestamp_str),
            filename=f'{experiment_timestamp_str}' + '-{epoch:02d}-{valid_loss:.2f}',
            save_top_k=2,
            mode='min',
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

    def __add__(self, other):
        return other + self.callback_list

    def append(self, item):
        assert isinstance(item, Callback)
        self.callback_list.append(item)

