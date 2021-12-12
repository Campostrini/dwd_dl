import datetime as dt
import calendar
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask.distributed import Client

from dwd_dl import cfg
import dwd_dl.dataset as ds
import dwd_dl.stats as stats
from dwd_dl.utils import year_month_tuple_list

if __name__ == "__main__":
    cfg.initialize2()

    client = Client()

    radolan_dataset = ds.H5Dataset(cfg.CFG.date_ranges, mode='r')

    radolan_stat = stats.RadolanSingleStat(radolan_dataset)

    start_year_month = dt.date(2018, 9, 1)  # attention! If mismatch with DATE_RANGES.yml it gets confusing.
    end_year_month = dt.date(2021, 10, 31)

    n_rows = 9
    n_cols = 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8.27, 11.69), sharex='row', sharey='col')

    def year_month_custom_periods(start_year_month_, end_year_month_):
        ym_tuple_list = year_month_tuple_list(start_year_month_, end_year_month_)
        out = []
        for year_, month_ in ym_tuple_list:
            out += [
                pd.date_range(
                    start=dt.datetime(
                        year=year_, month=month_, day=1, hour=0, minute=50
                    ), end=dt.datetime(
                        year=year_, month=month_, day=calendar.monthrange(year_, month_)[1], hour=23, minute=50
                    ), freq='H'
                )
            ]
        return out

    ym_list = year_month_tuple_list(start_year_month, end_year_month)
    ym_dict = {}
    for ym in ym_list:
        try:
            ym_dict[ym[0]].append(ym)
        except KeyError:
            ym_dict[ym[0]] = [ym]

    counter = 0
    for i in range(n_rows):
        for j in range(n_cols):
            # progressive numbering
            try:
                ym_annual_list = ym_dict[start_year_month.year + counter]
                ym_start = ym_annual_list[0]
                ym_end = ym_annual_list[-1]
                start_year = dt.date(year=ym_start[0], month=ym_start[1], day=1)
                end_year = dt.date(year=ym_end[0], month=ym_end[1], day=31)
                radolan_stat.boxplot(
                    ax=axs[i, j], custom_periods=year_month_custom_periods(start_year, end_year),  # have a check
                    xticklabels=[abbr[0] for abbr in list(calendar.month_abbr)[1:]], autorange=True,
                    positions=list(range(ym_start[1], ym_end[-1]+1)), auto_12_months=True
                )
                # axs[i, j].set(ylim=(-10, 400))
                axs[i, j].text(0.1, 0.9, f"{start_year_month.year + counter}",
                               horizontalalignment='left', verticalalignment='center', transform=axs[i, j].transAxes)
            except KeyError:
                axs[i, j].axis('off')
            counter += 1

    fig.show()

    for year in ym_dict:
        cols = 2
        rows = math.ceil(len(ym_dict[year]) / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(8.27, 11.69), sharex='row', sharey='col')
        start_ym = dt.date(*ym_dict[year][0], day=1)
        end_ym = dt.date(*ym_dict[year][-1], day=calendar.monthrange(*ym_dict[year][-1])[1])
        ym_custom_periods = year_month_custom_periods(start_ym, end_ym)
        for n, ax in enumerate(axs.reshape(-1)):
            ratio_days_gt_zero = radolan_stat.rainy_days_ratio([ym_custom_periods[n]], 0)
            ratio_tiles_gt_zero = radolan_stat.rainy_pixels_ratio([ym_custom_periods[n]], 0)
            radolan_stat.hist(ax=ax, custom_periods=[ym_custom_periods[n]], linewidth=2,
                              transformation=lambda x: np.log(x + 0.01), bins=200, range=[-3, 8],
                              condition=lambda x: x > 0)
            ax.text(0.1, 0.9, f"Perc of days > 0: {float(ratio_days_gt_zero[0].compute())*100:.1f}%\n"
                              f"Perc of tiles > 0: {float(ratio_tiles_gt_zero[0].compute())*100:.1f}%",
                    horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        fig.show()
