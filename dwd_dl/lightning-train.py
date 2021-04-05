import os
import warnings
import datetime as dt

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from dwd_dl import cfg
import dwd_dl.model as model
import dwd_dl.callbacks as callbacks
import dwd_dl.data_module as data_module
import dwd_dl as dl
from dwd_dl.cli import RadolanParser
from dwd_dl.video import VideoProducer
from dwd_dl import yaml_utils


def main(args):
    experiment_timestamp_str = dt.datetime.now().strftime(cfg.CFG.TIMESTAMP_DATE_FORMAT)
    unet = model.UNetLitModel(**vars(args), timestamp_string=experiment_timestamp_str)
    dm = data_module.RadolanDataModule(args.batch_size, args.workers, args.image_size)
    logger = TestTubeLogger(
        os.path.join(cfg.CFG.RADOLAN_ROOT, 'tt_logs'),
        experiment_timestamp_str,
    )
    callbacks_list = callbacks.CallbacksList(experiment_timestamp_str)
    trainer = Trainer.from_argparse_args(args, logger=logger, flush_logs_every_n_steps=100, callbacks=callbacks_list)
    trainer.fit(unet, dm)
    checkpoint_path = cfg.CFG.create_checkpoint_path_with_name(experiment_timestamp_str)
    trainer.save_checkpoint(checkpoint_path)
    dm.close()

    if args.video:
        video_trainer = Trainer.from_argparse_args(args)
        dm = data_module.VideoDataModule(args.batch_size, args.workers, args.image_size)
        producer = VideoProducer(video_trainer, unet, dm, args.video_mode, args.frame_rate)
        producer.produce()


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    parser = Trainer.add_argparse_args(parser)
    parser = model.UNetLitModel.add_model_specific_args(parser)
    parser = VideoProducer.add_video_specific_argparse_args(parser)
    args = parser.parse_args()
    main(args)
