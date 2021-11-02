import datetime as dt
import calendar

import pandas as pd
import matplotlib.pyplot as plt

from dwd_dl import cfg
import dwd_dl.dataset as ds
import dwd_dl.stats as stats

if __name__ == "__main__":
    cfg.initialize(skip_download=True)
    if ds.check_datasets_missing(cfg.CFG.date_ranges, classes=cfg.CFG.CLASSES, mode=cfg.CFG.MODE):
        cfg.initialize()
        ds.create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES, normal_ranges=cfg.CFG.date_ranges,
                     video_ranges=cfg.CFG.video_ranges, path_to_folder=cfg.CFG.RADOLAN_H5)

    radolan_dataset = ds.H5Dataset(cfg.CFG.date_ranges, mode='r')

    radolan_stat = stats.RadolanSingleStat(radolan_dataset)

    start_year = 2006
    end_year = 2020

    n_rows = 7
    n_cols = 2

    fig, axs = plt.subplots(n_rows, n_cols)

    def year_month_custom_periods(start_year_, end_year_):
        out = []
        for year in range(start_year_, end_year_ + 1):
            for month in range(1, 13):
                out += [
                    pd.date_range(
                        start=dt.datetime(
                            year=year, month=1, day=1, hour=0, minute=50
                        ), end=dt.datetime(
                            year=year, month=12, day=calendar.monthrange(year, month)[1], hour=23, minute=50
                        )
                    )
                ]
        return out

    for i in range(n_rows):
        for j in range(n_cols):
            radolan_stat.boxplot(
                ax=axs[i, j], custom_periods=year_month_custom_periods(start_year, end_year),
                xticklabels=list(calendar.month_abbr)[1:]
            )

    fig.show()
