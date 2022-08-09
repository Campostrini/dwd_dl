# Radolan Stat Abstract Class
import tracemalloc
import warnings
from abc import ABC, abstractmethod
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
from xarray.plot.utils import get_axis, label_from_attrs, _update_axes
import matplotlib.pyplot as plt
import dask
import dask.array
import dask.dataframe
from tqdm import tqdm

from . import cfg
from .dataset import H5Dataset, RadolanDataset, RadolanSubset
from dwd_dl import log
from .axes import RadolanAxes as RAxes


class RadolanStatAbstractClass(ABC):
# Hist property
    @abstractmethod
    def hist(self):
        pass

# Whisker preparation for plot
    @abstractmethod
    def boxplot(self):
        pass


class RadolanSingleStat(RadolanStatAbstractClass):
    def __init__(self, radolan_datset: RadolanDataset):
        self._radolan_dataset = radolan_datset

    def hist(self,
             season=None,
             custom_periods=None,
             transformation=None,
             condition=None,
             figsize=None,
             size=None,
             aspect=None,
             ax=None,
             xincrease=None,
             yincrease=None,
             xscale=None,
             yscale=None,
             xticks=None,
             yticks=None,
             xlim=None,
             ylim=None,
             xticklabels=None,
             bins=None,
             range=None,
             combine=False,
             title='',
             **kwargs
             ):
        """Adapted from xarray

        """
        log.info(f"Computing histogram for {custom_periods=}")
        if isinstance(custom_periods, list):
            dataarray = self._data_array_selection_from_custom_periods(season, custom_periods, combine)
        else:
            raise ValueError("Whhoops, something went wrong.")

        log.info(f"Gettin axis for histogram.")
        ax = get_axis(figsize, size, aspect, ax)
        arrays = []
        if not isinstance(dataarray, list):
            dataarray = [dataarray]

        for array in dataarray:
            if condition is not None:
                assert callable(condition)
                no_nan = dask.array.ravel(array.where(condition(array)))
            else:
                no_nan = dask.array.ravel(array)
            no_nan = dask.array.ma.masked_where(dask.array.isnan(no_nan), no_nan)
            arrays.append(no_nan)

        if transformation is not None:
            assert callable(transformation)
            log.info("Apply transformations.")
            arrays = [transformation(array) for array in arrays]

        for array in arrays:
            h, bins = dask.array.histogram(array[~dask.array.ma.getmaskarray(array)], bins=bins, range=range)
            bins = np.array(bins)
            h = np.array(h)
            print(h, bins)
            # primitive = ax.bar(bins[1:], h, **kwargs)
            primitive = ax.bar(bins[:-1], h, width=np.diff(bins), ec="k", align="edge",)

        ax.set_title("Histogram " + title)
        ax.set_xlabel("Precipitation")
        if xticklabels is not None:
            assert len(xticklabels) == len(custom_periods)
            ax.set_xticklabels(xticklabels)

        _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)

        return primitive

    def hist_results(self,
             season=None,
             custom_periods=None,
             transformation=None,
             condition=None,
             figsize=None,
             size=None,
             aspect=None,
             xincrease=None,
             yincrease=None,
             xscale=None,
             yscale=None,
             xticks=None,
             yticks=None,
             xlim=None,
             ylim=None,
             xticklabels=None,
             bins=None,
             range=None,
             combine=False,
             title='',
             **kwargs
             ):
        """Adapted from xarray

        """
        log.info(f"Computing histogram for {custom_periods=}")
        if isinstance(custom_periods, list):
            dataarray = self._data_array_selection_from_custom_periods(season, custom_periods, combine)
        else:
            raise ValueError("Whhoops, something went wrong.")

        log.info(f"Gettin axis for histogram results.")
        arrays = []
        if not isinstance(dataarray, list):
            dataarray = [dataarray]

        for array in dataarray:
            if condition is not None:
                assert callable(condition)
                no_nan = dask.array.ravel(array.where(condition(array)))
            else:
                no_nan = dask.array.ravel(array)
            no_nan = dask.array.ma.masked_where(dask.array.isnan(no_nan), no_nan)
            arrays.append(no_nan)

        if transformation is not None:
            assert callable(transformation)
            log.info("Apply transformations.")
            arrays = [transformation(array) for array in arrays]

        h_list = []
        bins_ = None
        log.info("Computing histogram in histogram results")
        for array in arrays:
            # h_, bins_ = dask.array.histogram(array[~dask.array.ma.getmaskarray(array)], bins=bins, range=range)
            h_, bins_ = dask.array.histogram(array, bins=bins, range=range)
            h_list.append(h_.compute())

        h = h_list[0]
        for h_ in h_list[1:]:
            for i, elem in enumerate(h_):
                h[i] += elem

        return h, bins_.compute()

    def scatter(self,
                season=None,
                custom_periods=None,
                transformation=None,
                condition=None,
                figsize=None,
                size=None,
                aspect=None,
                ax=None,
                xincrease=None,
                yincrease=None,
                xscale=None,
                yscale=None,
                xticks=None,
                yticks=None,
                xlim=None,
                ylim=None,
                xticklabels=None,
                range=None,
                combine=False,
                title='',
                plot=True,
                **kwargs
                ):
        """Adapted from xarray

        """
        log.info(f"Computing histogram for {custom_periods=}")
        if isinstance(custom_periods, list):
            dataarray = self._data_array_selection_from_custom_periods(season, custom_periods, combine)
        else:
            raise ValueError("Whhoops, something went wrong.")
        if plot:
            log.info(f"Gettin axis for scatter plot.")
            ax = get_axis(figsize, size, aspect, ax)
        arrays = []
        if not isinstance(dataarray, list):
            dataarray = [dataarray]

        for array in dataarray:
            if condition is not None:
                assert callable(condition)
                no_nan = dask.array.ravel(array.where(condition(array)))
            else:
                no_nan = dask.array.ravel(array)
            no_nan = dask.array.ma.masked_where(dask.array.isnan(no_nan), no_nan)
            arrays.append(no_nan)

        if transformation is not None:
            assert callable(transformation)
            log.info("Apply transformations.")
            arrays = [transformation(array) for array in arrays]


        value_counts_dicts = []
        log.info(f"Computing value counts for {arrays=}")
        for array in arrays:
            vc = dask.dataframe.from_array(array).value_counts()
            tracemalloc.take_snapshot()
            log.info(vc)
            x, y = vc.index.values, vc.values
            x = x.compute()
            y = y.compute()
            tracemalloc.take_snapshot()
            value_counts_dicts.append({key: value for key, value in zip(x, y)})

        value_counts = value_counts_dicts[0]
        for vcd in value_counts_dicts[1:]:
            for key in vcd:
                try:
                    value_counts[key] += vcd[key]
                except KeyError:
                    value_counts[key] = vcd[key]

        value_counts = dict(sorted(value_counts.items()))

        v_sum = sum(list(value_counts.values()))
        values = [v/v_sum for v in list(value_counts.values())]
        if plot:
            primitive = ax.scatter(list(value_counts.keys())[1:], values[1:])
        else:
            primitive = None
        if plot:
            ax.set_title(title)
            ax.set_xlabel("Precipitation [mm/h]")
        if xticklabels is not None:
            assert len(xticklabels) == len(custom_periods)
            ax.set_xticklabels(xticklabels)

        _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)

        return primitive, value_counts

#    @dask.delayed
    def boxplot(
        self,
        season=None,
        custom_periods=None,
        transformation=None,
        condition=None,
        figsize=None,
        size=None,
        aspect=None,
        ax=None,
        xincrease=None,
        yincrease=None,
        xscale=None,
        yscale=None,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        xticklabels=None,
        **kwargs,
    ):
        """Adapted from xarray

        """
        log.info(f"Computing boxplot for {custom_periods=}")
        if isinstance(custom_periods, list):
            dataarray = self._data_array_selection_from_custom_periods(season, custom_periods)
        else:
            raise ValueError("Whhoops, something went wrong.")

        ax = get_axis(figsize, size, aspect, ax)
        arrays = []

        for array in dataarray:
            if condition is not None:
                assert callable(condition)
                no_nan = dask.array.ravel(array.where(condition(array)))
            else:
                no_nan = dask.array.ravel(array)
            no_nan = dask.array.ma.masked_where(dask.array.isnan(no_nan), no_nan)
            arrays.append(no_nan)

        if transformation is not None:
            assert callable(transformation)
            arrays = [transformation(array) for array in arrays]

        if xticklabels is None:
            xticklabels = self._get_labels_from_periods(custom_periods)
        # else:
        #     assert len(xticklabels) == len(custom_periods)

        primitive = RAxes.static_boxplot(ax, arrays, **kwargs)

        ax.set_title("Precipitation boxplot")
        ax.set_xlabel(label_from_attrs(self._h5dataset.ds.precipitation).capitalize())
        if len(ax.get_xticks()) == 2*len(xticklabels):
            ax.set_xticklabels(2*xticklabels)
        else:
            ax.set_xticklabels(xticklabels)

        _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)

        return primitive

    @property
    def _data_array(self):
        return self._radolan_dataset.only_last_channel_dataset()

    @property
    def _winter_data_array(self):
        return self._radolan_dataset.only_last_channel_dataset().where(
            (self._radolan_dataset.only_last_channel_dataset()['time.month'] <= 4) |
            (self._radolan_dataset.only_last_channel_dataset()['time.month'] >= 10)
        )

    @property
    def _summer_data_array(self):
        return self._radolan_dataset.only_last_channel_dataset().where(
            (self._radolan_dataset.only_last_channel_dataset()['time.month'] >= 5) &
            (self._radolan_dataset.only_last_channel_dataset()['time.month'] <= 9)
        )

    def _get_data_array(self, season):
        if season == 'all':
            return self._data_array
        elif season == "summer":
            return self._summer_data_array
        elif season == "winter":
            return self._winter_data_array
        else:
            raise ValueError(f"Unrecognized value {season=}")

    def _data_array_selection_from_custom_periods(self, season, custom_periods, combine=False):
        dataarray = []
        for period in custom_periods:
            if isinstance(period, list):
                dataarray = self._data_array_selection_from_custom_periods(season, period, combine=combine)
            elif not isinstance(period, pd.DatetimeIndex):
                raise ValueError(f"Don't know what to do with element of type {type(period)} in custom_periods "
                                 f"{custom_periods}.")
            else:
                array_for_selection = self._get_data_array(season)
                dataarray += [
                    array_for_selection.where(array_for_selection.time.isin(period), drop=True)
                ]
        if combine:
            out = dataarray[0]
            for array in dataarray[1:]:
                out = out.combine_first(array)
            return out
        return dataarray
        # TODO: implement some checks on periods so that there is at least something to plot

    def rainy_pixels_ratio(self, custom_periods=None, threshold=0, compute=False):
        log.info("Computing rainiy pixels ratio.")
        th = threshold
        if custom_periods is None:
            out = [(self._data_array > th).sum() / self._data_array.size]
        else:
            assert isinstance(custom_periods, list)
            for period in custom_periods:
                assert isinstance(period, pd.DatetimeIndex)
            out = [
                (sliced := self._data_array.where(
                    self._data_array.time.isin(period), drop=True
                ) > th).sum()/sliced.size
                for period in custom_periods
            ]
        if compute:
            out = [element.compute().values for element in out]
        log.info("Done computing rainy pixels ratio.")
        return out

    def rainy_days_ratio(self, custom_periods=None, threshold=0, compute=False):
        log.info("Computing rainy days ratio.")
        th = threshold
        if custom_periods is None:
            out = [(greater_zero_days := ((self._data_array > th).sum(dim=['lon', 'lat']) > th)).sum() / len(
                greater_zero_days
            )]
        else:
            assert isinstance(custom_periods, list)
            for period in custom_periods:
                assert isinstance(period, pd.DatetimeIndex)
            out = [
                (sliced := (self._data_array.where(
                    self._data_array.time.isin(period), drop=True
                ).sum(dim=['lon', 'lat']) > th)).sum()/len(sliced)
                for period in custom_periods
            ]
        if compute:
            out = [element.compute().values for element in out]
        log.info("Done computing rainy days ratio.")
        return out

    def number_of_days_over_threshold(self, custom_periods=None, threshold=0):
        th = threshold
        if custom_periods is None:
            out = {'all': ((self._data_array > th).sum(dim=['lon', 'lat']) > th).sum()}
        else:
            assert isinstance(custom_periods, list)
            for period in custom_periods:
                assert isinstance(period, pd.DatetimeIndex)
            out = {
                period.min().to_pydatetime().date(): ((self._data_array.where(
                    self._data_array.time.isin(period), drop=True
                )).sum(dim=['lon', 'lat']) > th).sum()
                for period in custom_periods
            }
        return {element: int(out[element].compute().values) for element in out}

    def _get_labels_from_periods(self, periods=None):
        def label_from_start_end(start, end):
            template = "From {} to {}"
            return [template.format(start, end)]

        if periods is None:
            min_, max_ = self._data_array.time.min(), self._data_array.time.max()
            if not isinstance(min_, pd.Timestamp) and not isinstance(min_, pd.Timestamp):
                min_, max_ = pd.Timestamp(min_.values), pd.Timestamp(max_.values)
                return label_from_start_end(min_, max_)
        assert isinstance(periods, list)
        assert all([isinstance(period, pd.DatetimeIndex) for period in periods])

        return [label_from_start_end(period.min(), period.max()) for period in periods]

    def __add__(self, other):
        assert isinstance(other, RadolanSingleStat)
        raise NotImplementedError


class ClassCounter:
    def __init__(self, radolan_dataset: RadolanDataset):
        self._radolan_dataset = radolan_dataset
        self._count_class_out = {}
        self._dataset = self._radolan_dataset.only_last_channel_dataset_classes()

    def count_class(self, class_index: int, custom_periods=None, threshold=None):
        log.info(
            f"Counting class {class_index}"
        )
        if custom_periods is None:
            out = {'all': ((self._dataset == class_index).sum(dim=['lon', 'lat'])).sum()}
        else:
            assert isinstance(custom_periods, list)
            for period in custom_periods:
                assert isinstance(period, pd.DatetimeIndex)
            out = {
                period.min().to_pydatetime().date(): (self._dataset.where(
                    self._dataset.time.isin(period), drop=True
                ) == class_index).sum(dim=['lon', 'lat']).sum() for period in custom_periods
            }
        log.info(f"Finished counting class {class_index}")
        return {element: int(out[element].compute()) for element in out}

    def count_all_classes(self, custom_periods=None, threshold=None):
        out = {}
        for n, c in enumerate(cfg.CFG.CLASSES):
            out[c] = self.count_class(class_index=n, custom_periods=custom_periods, threshold=threshold)
        return out

    @staticmethod
    def sum_on_all_periods(classes_dict_out):
        out = {}
        for c in classes_dict_out:
            out[c] = sum(classes_dict_out[c].values())
        return out

    @property
    def classes(self):
        return cfg.CFG.CLASSES

