import os
import warnings
from dwd_dl import cfg
import dwd_dl.model as model
import dwd_dl as dl
from dwd_dl.cli import RadolanParser
from dwd_dl import yaml_utils
from pytorch_lightning import Trainer


def main(parser_args, **kwargs):
    unet = model.UNetLitModel(**kwargs)
    trainer = Trainer.from_argparse_args(parser_args)
    trainer.fit(unet)

def makedirs(weights, logs):
    os.makedirs(weights, exist_ok=True)
    os.makedirs(logs, exist_ok=True)


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    parser = Trainer.add_argparse_args(parser)
    parser_args = parser.parse_args()
    kwargs_ = vars(parser_args)
    if 'filename' in kwargs_:
        warnings.warn("The --filename option is no longer supported. It will be ignored.", DeprecationWarning)
        del kwargs_['filename']
    # yaml_utils.log_dump(**kwargs_)
    main(parser_args=parser_args, **kwargs_)