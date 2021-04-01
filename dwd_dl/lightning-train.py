import os
import warnings
import datetime as dt

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from dwd_dl import cfg
import dwd_dl.model as model
import dwd_dl.callbacks as callbacks
import dwd_dl as dl
from dwd_dl.cli import RadolanParser
from dwd_dl import yaml_utils


def main(args):
    unet = model.UNetLitModel(**vars(args))
    experiment_timestamp_str = dt.datetime.now().strftime(cfg.CFG.TIMESTAMP_DATE_FORMAT)
    logger = TestTubeLogger(
        os.path.join(cfg.CFG.RADOLAN_ROOT, 'tt_logs'),
        experiment_timestamp_str,
    )
    callbacks_list = callbacks.CallbacksList(experiment_timestamp_str)
    trainer = Trainer.from_argparse_args(args, logger=logger, flush_logs_every_n_steps=5, callbacks=callbacks_list)
    trainer.fit(unet)
    checkpoint_path = cfg.CFG.create_checkpoint_path(experiment_timestamp_str)
    trainer.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    parser = Trainer.add_argparse_args(parser)
    parser = model.UNetLitModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
