import datetime as dt
import calendar
import math
import os.path
from typing import List, Iterable
import tracemalloc

import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from dask.distributed import Client
import dask.config

from dwd_dl import cfg
from dwd_dl import log
import dwd_dl.dataset as ds
import dwd_dl.stats as stats
from dwd_dl.utils import year_month_tuple_list

if __name__ == "__main__":
    tracemalloc.start()
    cfg.initialize2()

    figures_path = os.path.join(os.path.abspath(cfg.CFG.RADOLAN_ROOT), 'figures')

    client = Client(memory_limit='68G')

    # radolan_dataset = ds.H5Dataset(cfg.CFG.date_ranges, mode='r')
    radolan_dataset = ds.RadolanDataset()
    # radolan_dataset_classes = ds.H5Dataset(cfg.CFG.date_ranges, mode='c')

    radolan_stat = stats.RadolanSingleStat(radolan_dataset)
    class_counter = stats.ClassCounter(radolan_dataset)

    start_year_month = dt.date(2020, 9, 1)  # attention! If mismatch with DATE_RANGES.yml it gets confusing.
    end_year_month = dt.date(2021, 12, 31)

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

    def group_by_year(list_of_date_ranges: List[DatetimeIndex]):
        group_dict = {}
        for range_ in list_of_date_ranges:
            try:
                group_dict[range_[0].year].append(range_)
            except KeyError:
                group_dict[range_[0].year] = range_
        return list(group_dict.values())

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
            self.date_range_winter = group_by_year(self.date_range_winter)
            self.date_range_summer = exclude_months_not_in_given_tuple(
                self.date_range, PeriodForPlot.summer_months
            )
            self.date_range_summer = group_by_year(self.date_range_summer)
            self.date_range = group_by_year(self.date_range)
            self.name = name

        def __iter__(self):
            return iter(
                [('all', self.date_range),
                 ('winter', self.date_range_winter),
                 ('summer', self.date_range_summer)]
            )

    def rainy_timestamps_format(period_, sub_period_name_, sub_period_, threshold, radolan_stat_: stats.RadolanSingleStat):
        out = f"Rainy timestamps in {period_.name} ({sub_period_name_} months) are "
        out += f"{sum(radolan_stat_.number_of_days_over_threshold(sub_period_, threshold).values())}"
        out += f"\n over a total of {sum([len(cp) for cp in sub_period_])} timestamps."
        return out

    def rainy_classes_format(class_counter_: stats.ClassCounter, sub_period_, threshold=None):
        counted_classes = class_counter_.sum_on_all_periods(class_counter.count_all_classes(
            sub_period_, threshold=threshold))
        out = f"The classes are subdivided in "
        out += f"{counted_classes}"
        out += f"\nover a total of {sum([len(cp) for cp in sub_period_]) * 256 * 256} tiles."
        out += f"\nThe sum of all classes tiles is {sum(counted_classes.values())}"
        return out

    short = False
    if short:
        ranges_list = [
            {
                'name': 'Training + Validation + Test',
                'start': dt.date(2019, 10, 1),
                'end': dt.date(2021, 12, 31)
            },
            {
                'name': 'Training',
                'start': dt.date(2019, 10, 1),
                'end': dt.date(2020, 4, 30)
            },
            {
                'name': 'Validation',
                'start': dt.date(2020, 10, 1),
                'end': dt.date(2021, 4, 30)
            },
            {
                'name': 'Test',
                'start': dt.date(2021, 10, 1),
                'end': dt.date(2021, 12, 31)
            }
        ]

    else:
        ranges_list = [
            {
                'name': 'Training + Validation + Test',
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

    scatter = True
    if scatter:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            for period in period_list_for_plot:
                for sub_period_name, sub_period in period:
                    if not sub_period:
                        continue
                    fig, axs = plt.subplots(1, 1, figsize=(8.27, 11.69))
                    ax = axs
                    custom_period = sub_period[0]
                    for p in sub_period[1:]:
                        custom_period = custom_period.append(p)
                    try:
                        tracemalloc.take_snapshot()
                        title = f"{period.name}, {sub_period_name}"
                        radolan_stat.scatter(ax=ax, custom_periods=sub_period, linewidth=2, season=sub_period_name,
                                             bins=np.logspace(np.log(0.09), np.log(20), 50, base=np.e),
                                             title=title, combine=False)
                        tracemalloc.take_snapshot()
                    except Exception as exc:
                        log.info(exc)
                        continue
                                            # transformation=lambda x: np.log(x + 0.01),
                                            # bins=200, range=[-3, 8], condition=lambda x: x > 0,
                                            # title=f"{period.name}, {sub_period_name}", combine=True)
                    ratio_days_gt_zero = radolan_stat.rainy_days_ratio([custom_period], 0)
                    ratio_tiles_gt_zero = radolan_stat.rainy_pixels_ratio([custom_period], 0)
                    ax.text(0.1, 0.9, f"Perc of days > 0: {float(ratio_days_gt_zero[0].compute())*100:.1f}%\n"
                                      f"Perc of tiles > 0: {float(ratio_tiles_gt_zero[0].compute())*100:.1f}%",
                            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                    ax.set_xscale('log')
                    rainy_days_string = rainy_timestamps_format(period, sub_period_name, [custom_period],
                                                                threshold=0, radolan_stat_=radolan_stat)
                    rainy_classes_string = rainy_classes_format(class_counter, [custom_period], threshold=0)
                    log.info(rainy_days_string)
                    log.info(rainy_classes_string)
                    plt.savefig(os.path.join(figures_path, title + str(dt.datetime.now()) + ".png"))

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        for period in period_list_for_plot:
            for sub_period_name, sub_period in period:
                if not sub_period:
                    continue
                custom_period = sub_period[0]
                for p in sub_period[1:]:
                    custom_period = custom_period.append(p)

                try:
                    h, bins = radolan_stat.hist_results(custom_periods=sub_period, season=sub_period_name,
                                                        bins=[0, 0.1, 1.0, 2.5, 500], combine=True)
                except Exception as exc:
                    log.info(exc)
                log.info(f"{period.name=}, {sub_period_name=}")
                log.info(f"h={str(h)} \n bins={str(bins)}")


    log_histogram = False
    if log_histogram:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            for period in period_list_for_plot:
                for sub_period_name, sub_period in period:
                    if not sub_period:
                        continue
                    fig, axs = plt.subplots(1, 1, figsize=(8.27, 11.69))
                    ax = axs

                    custom_period = sub_period[0]
                    for p in sub_period[1:]:
                        custom_period = custom_period.append(p)
                    try:
                        radolan_stat.hist(ax=ax, custom_periods=[custom_period], linewidth=2, season=sub_period_name,
                                          # bins=np.logspace(np.log(0.01), np.log(300), num=200, base=np.e), transformation=lambda x: x + 0.01,
                                          bins=np.logspace(np.log(0.09), np.log(20), 50, base=np.e), transformation= lambda x: x + 0.01,
                                          title=f"{period.name}, {sub_period_name}", combine=True)
                    except Exception as exc:
                        print(exc)
                        continue
                                      # transformation=lambda x: np.log(x + 0.01),
                                      # bins=200, range=[-3, 8], condition=lambda x: x > 0,
                                      # title=f"{period.name}, {sub_period_name}", combine=True)

                    ratio_days_gt_zero = radolan_stat.rainy_days_ratio([custom_period], 0)
                    ratio_tiles_gt_zero = radolan_stat.rainy_pixels_ratio([custom_period], 0)
                    ax.text(0.1, 0.9, f"Perc of days > 0: {float(ratio_days_gt_zero[0].compute())*100:.1f}%\n"
                                      f"Perc of tiles > 0: {float(ratio_tiles_gt_zero[0].compute())*100:.1f}%",
                            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                    ax.set_xscale('log')
                    rainy_days_string = rainy_timestamps_format(period, sub_period_name, [custom_period],
                                                                threshold=0, radolan_stat_=radolan_stat)
                    rainy_classes_string = rainy_classes_format(class_counter, [custom_period], threshold=0)
                    print(rainy_days_string)
                    print(rainy_classes_string)
                    fig.show()




