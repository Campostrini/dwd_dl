from dwd_dl import cfg
from dwd_dl import dataset as ds


if __name__ == "__main__":
    cfg.initialize(skip_download=True)
    if ds.check_datasets_missing(cfg.CFG.date_ranges, classes=cfg.CFG.CLASSES):
        cfg.initialize()
        ds.create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES, normal_ranges=cfg.CFG.date_ranges,
                     video_ranges=cfg.CFG.video_ranges, path_to_folder=cfg.CFG.RADOLAN_H5)
