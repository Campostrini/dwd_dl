import datetime as dt

import torch
from dask.distributed import Client

import dwd_dl.model as model
import dwd_dl.data_module as data_module
import dwd_dl as dl
import dwd_dl.img as img
from dwd_dl.cli import RadolanParser

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
matplotlib.use('WebAgg')
import cmasher as cmr


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
            fig = plt.figure(figsize=(11.69, 8.27))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(2, 3),
                             axes_pad=0.15,
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="7%",
                             cbar_pad=0.15)
            cmap = cmr.get_sub_cmap('coolwarm_r', 0.6, 1)
            for images in series[0]:
                for axs, image in zip(grid, images):
                    im = axs.imshow(image, cmap=cmap, vmin=0, vmax=20)

            axs.cax.colorbar(im)
            axs.cax.toggle_label(True)

            plt.savefig(f'/home/stefano/Desktop/image_{timestamp}_input.png')

            cm = matplotlib.cm.get_cmap('coolwarm')
            cmap = colors.ListedColormap([cm(0.4), cm(0.3), cm(0.2), cm(0.1)])
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            fig = plt.figure(figsize=(11.69, 8.27))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(1, 2),
                             axes_pad=0.15,
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="7%",
                             cbar_pad=0.15)
            grid[0].imshow(series[1][0][0], cmap=cmap, vmin=-0.5, vmax=3.5)
            im = grid[1].imshow(series[2][0][0], cmap=cmap, vmin=-0.5, vmax=3.5)
            cbar = grid[1].cax.colorbar(im, cmap=cmap, boundaries=bounds, norm=norm)
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(['0', '0.1', '1', '2.5'])
            plt.savefig(f'/home/stefano/Desktop/image_{timestamp}_output.png')


if __name__ == "__main__":
    client = Client()
    dl.cfg.initialize2()
    parser = RadolanParser()
    parser = model.RadolanLiveEvaluator.add_model_specific_args(parent_parser=parser)
    args = parser.parse_args()
    # yaml_utils.log_dump(**kwargs_)
    main(args)
