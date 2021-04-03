from pytorch_lightning import Trainer
import torch

import dwd_dl.model as model
import dwd_dl.data_module as data_module
import dwd_dl as dl
from dwd_dl.cli import RadolanParser


def main(args):
    unet = model.UNetLitModel(**vars(args))
    dm = data_module.RadolanDataModule(args.batch_size, args.workers, args.image_size)

    trainer = Trainer.from_argparse_args(args)
    if args.model_path.endswith('.ckpt'):
        unet.load_from_checkpoint(checkpoint_path=args.model_path)
    elif args.model_path.endswith('.pt'):
        unet.load_state_dict(torch.load(args.model_path))
    elif args.model_path is not None:
        raise ValueError(f"Don't know what to do with {args.model_path}")
    else:
        print("Nothing to do")
        return
    prediction = trainer.predict(unet, datamodule=dm)
    print(prediction)


if __name__ == "__main__":
    dl.cfg.initialize()
    parser = RadolanParser()
    parser = Trainer.add_argparse_args(parser)
    parser = model.UNetLitModel.add_model_specific_args(parser)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="The path to the saved model."
    )
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
