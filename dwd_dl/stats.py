# Radolan Stat Abstract Class
import warnings
from abc import ABC, abstractmethod
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
from xarray.plot.utils import get_axis, label_from_attrs, _update_axes
import matplotlib.pyplot as plt

from . import cfg
from .dataset import H5Dataset


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
    def __init__(self, h5dataset: H5Dataset):
        self._h5dataset = h5dataset

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
             **kwargs
             ):
        """Adapted from xarray

        """
        if isinstance(custom_periods, list):
            dataarray = self._data_array_selection_from_custom_periods(season, custom_periods)
        else:
            raise ValueError("Whhoops, something went wrong.")

        ax = get_axis(figsize, size, aspect, ax)
        arrays = []
        if condition is not None:
            assert callable(condition)
        else:
            def condition(x): return x

        for array in dataarray:
            no_nan = np.ravel(array.where(condition(array)).to_numpy())
            no_nan = no_nan[pd.notnull(no_nan)]
            arrays.append(no_nan)

        if transformation is not None:
            assert callable(transformation)
            arrays = [transformation(array) for array in arrays]

        labels = self._get_labels_from_periods(custom_periods)

        primitive = ax.hist(arrays, **kwargs)

        ax.set_title(dataarray[0]._title_for_slice())
        ax.set_xlabel("Precipitation")
        ax.set_xticklabels(labels)

        _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)

        return primitive

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
        **kwargs,
    ):
        """Adapted from xarray

        """
        if isinstance(custom_periods, list):
            dataarray = self._data_array_selection_from_custom_periods(season, custom_periods)
        else:
            raise ValueError("Whhoops, something went wrong.")

        ax = get_axis(figsize, size, aspect, ax)
        arrays = []
        if condition is not None:
            assert callable(condition)
        else:
            def condition(x): return x

        for array in dataarray:
            no_nan = np.ravel(array.where(condition(array)).to_numpy())
            no_nan = no_nan[pd.notnull(no_nan)]
            arrays.append(no_nan)

        if transformation is not None:
            assert callable(transformation)
            arrays = [transformation(array) for array in arrays]

        labels = self._get_labels_from_periods(custom_periods)

        primitive = ax.boxplot(arrays, positions=range(len(arrays)), **kwargs)

        ax.set_title("Precipitation boxplot")
        ax.set_xlabel(label_from_attrs(self._h5dataset.ds.precipitation).capitalize())
        ax.set_xticklabels(labels)

        _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)

        return primitive

    @property
    def _data_array(self):
        return self._h5dataset.ds.precipitation

    @property
    def _winter_data_array(self):
        return self._h5dataset.ds.precipitation.where(
            (self._h5dataset.ds.precipitation['time.month'] <= 4) |
            (self._h5dataset.ds.precipitation['time.month'] >= 10)
        )

    @property
    def _summer_data_array(self):
        return self._h5dataset.ds.precipitation.where(
            (self._h5dataset.ds.precipitation['time.month'] >= 5) &
            (self._h5dataset.ds.precipitation['time.month'] <= 9)
        )

    def _get_data_array(self, season):
        if season is None:
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
                dataarray = self._data_array_selection_from_custom_periods(season, period, combine=True)
            elif not isinstance(period, pd.DatetimeIndex):
                raise ValueError(f"Don't know what to do with element of type {type(period)} in custom_periods "
                                 f"{custom_periods}.")
            else:
                array_for_selection = self._get_data_array(season)
                dataarray += [
                    array_for_selection.sel(time=slice(period.min().to_datetime64(), period.max().to_datetime64()))
                ]
        if combine:
            out = dataarray[0]
            for array in dataarray[1:]:
                out = out.combine_first(array)
            return out
        return dataarray
        # TODO: implement some checks on periods so that there is at least something to plot

    def rainy_pixels_ratio(self, custom_periods=None, threshold=0, compute=False):
        th = threshold
        if custom_periods is None:
            out = [(self._data_array > th).sum() / self._data_array.size]
        else:
            assert isinstance(custom_periods, list)
            for period in custom_periods:
                assert isinstance(period, pd.DatetimeIndex)
            out = [
                (sliced := self._data_array.sel(
                    time=slice(
                        period.min().to_datetime64(),
                        period.max().to_datetime64())
                )).sum()/sliced.size
                for period in custom_periods
            ]
        if compute:
            out = [element.compute().values for element in out]

        return out

    def rainy_days_ratio(self, custom_periods=None, threshold=0, compute=False):
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
                (sliced := ((self._data_array.sel(
                    time=slice(
                        period.min().to_datetime64(),
                        period.max().to_datetime64())
                )).sum(dim=['lon', 'lat']) > th)).sum()/len(sliced)
                for period in custom_periods
            ]
        if compute:
            out = [element.compute().values for element in out]

        return out

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
