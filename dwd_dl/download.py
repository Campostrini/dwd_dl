from dwd_dl import cfg
from dwd_dl import dataset as ds


if __name__ == "__main__":
    cfg.initialize()
    ds.create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)
