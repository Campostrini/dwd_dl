from dwd_dl import cfg
from dwd_dl import dataset as ds


if __name__ == "__main__":
    cfg.initialize(skip_download=True)
    if ds.check_datasets_missing(cfg.CFG.date_ranges, classes=cfg.CFG.CLASSES):
        cfg.initialize()
        ds.create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)
