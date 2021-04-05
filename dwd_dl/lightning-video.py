import datetime as dt

from pytorch_lightning import Trainer
import torch

import dwd_dl.model as model
import dwd_dl.data_module as data_module
import dwd_dl as dl
from dwd_dl.cli import RadolanParser
from dwd_dl.video import VideoProducer


def main(args):
    timestamp_string = dt.datetime.now().strftime(dl.cfg.CFG.TIMESTAMP_DATE_FORMAT)
    unet = model.UNetLitModel(**vars(args))
    dm = data_module.VideoDataModule(args.batch_size, args.workers, args.image_size)

    trainer = Trainer.from_argparse_args(args)
    if args.model_path is not None:
        if args.model_path.endswith('.ckpt'):
            unet.load_from_checkpoint(checkpoint_path=args.model_path)
            unet.assign_timestamp_string_from_checkpoint_path(checkpoint_path=args.model_path)
        elif args.model_path.endswith('.pt'):
            unet.load_state_dict(torch.load(args.model_path))
        else:
            raise ValueError(f"Don't know what to do with {args.model_path}")
    else:
        print("Nothing to do")
        return

    if unet.timestamp_string is None:
        unet.timestamp_string = timestamp_string

    producer = VideoProducer(trainer, unet, dm, args.video_mode, args.frame_rate)
    producer.produce()


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    parser = Trainer.add_argparse_args(parser)
    parser = model.UNetLitModel.add_model_specific_args(parser)
    parser = VideoProducer.add_video_specific_argparse_args(parser)
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
