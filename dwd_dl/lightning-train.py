import os
import warnings
import datetime as dt

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from dwd_dl import cfg
import dwd_dl.model as model
import dwd_dl as dl
from dwd_dl.cli import RadolanParser
from dwd_dl import yaml_utils


def main(args):
    unet = model.UNetLitModel(**vars(args))
    logger = TestTubeLogger(
        os.path.join(dl.cfg.CFG.RADOLAN_ROOT, 'tt_logs'),
        dt.datetime.now().strftime(dl.cfg.CFG.TIMESTAMP_DATE_FORMAT),
    )
    trainer = Trainer.from_argparse_args(args, logger=logger, flush_logs_every_n_steps=5)
    trainer.fit(unet)


def makedirs(weights, logs):
    os.makedirs(weights, exist_ok=True)
    os.makedirs(logs, exist_ok=True)


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    parser = Trainer.add_argparse_args(parser)
    parser = model.UNetLitModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
