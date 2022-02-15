import datetime as dt
import calendar
import math
from typing import List, Iterable

import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import matplotlib.pyplot as plt
from dask.distributed import Client
import dask.config

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


    only_winter_and_summer = True

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

    def exclude_months_not_in_given_tuple(list_of_date_ranges: List[DatetimeIndex], inclusion: Iterable[int]):
        return [range_ for range_ in list_of_date_ranges if range_[0].month in inclusion]

    ym_list = year_month_tuple_list(start_year_month, end_year_month)
    ym_dict = {}
    for ym in ym_list:
        try:
            ym_dict[ym[0]].append(ym)
        except KeyError:
            ym_dict[ym[0]] = [ym]

    counter = 0

    if not only_winter_and_summer:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8.27, 11.69), sharex='row', sharey='col')
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

    if not only_winter_and_summer:
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

    class PeriodForPlot:
        summer_months = (5, 6, 7, 8, 9)
        winter_months = (1, 2, 3, 4, 10, 11, 12)

        def __init__(self, start: dt.date, end: dt.date, name: str):
            self.start = start
            self.end = end
            self.date_range = year_month_custom_periods(self.start, self.end)
            self.date_range_winter = exclude_months_not_in_given_tuple(
                self.date_range, PeriodForPlot.winter_months
            )
            self.date_range_summer = exclude_months_not_in_given_tuple(
                self.date_range, PeriodForPlot.summer_months
            )
            self.name = name

        def __iter__(self):
            return iter(
                [('all', self.date_range),
                 ('winter', self.date_range_winter),
                 ('summer', self.date_range_summer)]
            )

    ranges_list = [
        {
            'name': 'Entire Period',
            'start': dt.date(2005, 10, 1),
            'end': dt.date(2021, 12, 31)
        },
        {
            'name': 'Training',
            'start': dt.date(2005, 10, 1),
            'end': dt.date(2017, 12, 31)
        },
        {
            'name': 'Validation',
            'start': dt.date(2018, 1, 1),
            'end': dt.date(2019, 12, 31)
        },
        {
            'name': 'Test',
            'start': dt.date(2020, 1, 1),
            'end': dt.date(2021, 12, 31)
        }
    ]

    period_list_for_plot = [PeriodForPlot(**range_) for range_ in ranges_list]

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        for period in period_list_for_plot:
            for sub_period_name, sub_period in period:
                fig, axs = plt.subplots(1, 1, figsize=(8.27, 11.69))
                ax = axs
                ratio_days_gt_zero = radolan_stat.rainy_days_ratio(sub_period, 0)
                ratio_tiles_gt_zero = radolan_stat.rainy_pixels_ratio(sub_period, 0)
                custom_period = sub_period[0]
                for p in sub_period[1:]:
                    custom_period.append(p)
                radolan_stat.hist(ax=ax, custom_periods=[custom_period], linewidth=2, transformation=lambda x: np.log(x + 0.01),
                                  bins=200, range=[-3, 8], condition=lambda x: x > 0,
                                  title=f"{period.name}, {sub_period_name}", combine=True)
                ax.text(0.1, 0.9, f"Perc of days > 0: {float(ratio_days_gt_zero[0].compute())*100:.1f}%\n"
                                  f"Perc of tiles > 0: {float(ratio_tiles_gt_zero[0].compute())*100:.1f}%",
                        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                print(f"Rainy days in {period.name}, {sub_period_name} are "
                      f"{radolan_stat.number_of_days_over_threshold(sub_period, 0)}")
                fig.show()
