import datetime as dt

import torch
from dask.distributed import Client

import dwd_dl.model as model
import dwd_dl.data_module as data_module
import dwd_dl as dl
import dwd_dl.img as img
from dwd_dl.cli import RadolanParser

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')


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
        series1 = unet.on_timestamp(
            timestamp=dt.datetime.strptime(args.timestamp1, dl.cfg.CFG.TIMESTAMP_DATE_FORMAT))
        series2 = unet.on_timestamp(
            timestamp=dt.datetime.strptime(args.timestamp2, dl.cfg.CFG.TIMESTAMP_DATE_FORMAT))
        for series, timestamp in zip((series1, series2), (args.timestamp1, args.timestamp2)):
            fig, ax = plt.subplots(2, 6)
            for images in series[0]:
                for axs, image in zip(ax[0], images):
                    axs.imshow(image)

            ax[1, 2].imshow(series[1][0][0])
            ax[1, 3].imshow(series[2][0][0])

            plt.savefig('/home/stefano/Desktop/image.png')


if __name__ == "__main__":
    client = Client()
    dl.cfg.initialize2()
    parser = RadolanParser()
    parser = model.RadolanLiveEvaluator.add_model_specific_args(parent_parser=parser)
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
