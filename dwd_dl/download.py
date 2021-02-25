from dwd_dl import cfg
from dwd_dl import dataset as ds


if __name__ == "__main__":
    cfg.initialize(skip_download=True)
    if ds.check_h5_missing_or_corrupt(cfg.CFG.date_ranges, classes=cfg.CFG.CLASSES, mode=cfg.CFG.MODE):
        cfg.initialize()
        ds.create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)
