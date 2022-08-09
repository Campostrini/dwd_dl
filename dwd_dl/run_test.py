import os
import datetime as dt

import torch
from dask.distributed import Client
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer

import dwd_dl.model as model
import dwd_dl.data_module as data_module
import dwd_dl as dl
import dwd_dl.img as img
from dwd_dl.cli import RadolanParser
import dwd_dl.cfg as cfg
import dwd_dl.callbacks as callbacks


def main(args):
    client = Client() #processes=False)
    experiment_timestamp_str = dt.datetime.now().strftime(cfg.CFG.TIMESTAMP_DATE_FORMAT)
    unet = model.UNetLitModel(**vars(args), timestamp_string=experiment_timestamp_str)
    print(f"{args.dask=}")
    dm = data_module.RadolanDataModule(args.batch_size, args.workers, args.image_size, args.dask, client=client)
    dm.prepare_data()
    dm.setup()
    if args.model_path is not None:
        if args.model_path.endswith('.ckpt'):
            unet = model.RadolanLiveEvaluator.load_from_checkpoint(checkpoint_path=args.model_path, dm=dm)
        elif args.model_path.endswith('.pt'):
            unet.load_state_dict(torch.load(args.model_path))
    logger = TestTubeLogger(
        os.path.join(cfg.CFG.RADOLAN_ROOT, 'tt_logs'),
        experiment_timestamp_str,
        create_git_tag=True,
    )
    callbacks_list = callbacks.CallbacksList(experiment_timestamp_str)
    if args.max_epochs is None:
        args.max_epochs = 100
    trainer = Trainer.from_argparse_args(args, logger=logger,
                                         flush_logs_every_n_steps=100, callbacks=list(callbacks_list))

    trainer.test(unet, dm)


if __name__ == "__main__":
    client = Client()
    dl.cfg.initialize2()
    parser = RadolanParser()
    parser = Trainer.add_argparse_args(parser)
    parser = model.UNetLitModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
