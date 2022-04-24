import torch
from dask.distributed import Client

import dwd_dl.model as model
import dwd_dl.data_module as data_module
import dwd_dl as dl
import dwd_dl.img as img
from dwd_dl.cli import RadolanParser


def main(args):
    dm = data_module.RadolanLiveDataModule(args.batch_size, args.workers, args.image_size, args.dask)
    dm.prepare_data()
    dm.setup()
    unet = model.RadolanLiveEvaluator(dm, **vars(args))
    if args.model_path is not None:
        if args.model_path.endswith('.ckpt'):
            unet = model.RadolanLiveEvaluator.load_from_checkpoint(checkpoint_path=args.model_path, dm=dm)
        elif args.model_path.endswith('.pt'):
            unet.load_state_dict(torch.load(args.model_path))

    unet.eval()
    # print(f"Timestamps over threshold: {dm.timestamps_over_threshold}")
    with torch.no_grad():
        unet.to('cuda')
        img.visualizer(unet)


if __name__ == "__main__":
    client = Client()
    dl.cfg.initialize2()
    parser = RadolanParser()
    parser = model.RadolanLiveEvaluator.add_model_specific_args(parent_parser=parser)
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
