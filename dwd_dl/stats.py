# Radolan Stat Abstract Class
import warnings
from abc import ABC, abstractmethod
import datetime as dt

import numpy as np
import pandas as pd
import xarray.plot
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

    def hist(self, *args, period=None, **kwargs):
        if period is None:
            data_array = self._data_array
        elif period == 'summer':
            data_array = self._summer_data_array
        elif period == 'winter':
            data_array = self._winter_data_array
        else:
            raise ValueError(f"period: {period} argument not recognized.")
        xarray.plot.hist(data_array, *args, **kwargs)

    def boxplot(
        self,
        period=None,
        custom_periods=None,
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
        if period and custom_periods:
            raise ValueError(f"period and custom_periods are mutually exclusive. Got {period} and {custom_periods}")

        if period == 'summer':
            dataarray = self._summer_data_array
        elif period == 'winter':
            dataarray = self._winter_data_array
        elif period is not None:
            raise ValueError(f"Unexpected input for period: {period}. Either 'summer' or 'winter'.")
        elif isinstance(custom_periods, list):
            dataarray = []
            for period in custom_periods:
                if not isinstance(custom_periods, pd.DatetimeIndex):
                    raise ValueError(f"Don't know what to do with element of type {type(period)} in custom_periods "
                                     f"{custom_periods}.")
                dataarray += [self._data_array.where(self._data_array['time'] in period)]
                # TODO: continue implementation
        elif period is None and custom_periods is None:
            dataarray = self._data_array
        else:
            raise ValueError("Whhoops, something went wrong.")


        ax = get_axis(figsize, size, aspect, ax)
        no_nan = np.ravel(dataarray.to_numpy())
        no_nan = no_nan[pd.notnull(no_nan)]

        primitive = ax.boxplot(no_nan, **kwargs)

        ax.set_title(self._h5dataset.ds.precipitation._title_for_slice())
        ax.set_xlabel(label_from_attrs(self._h5dataset.ds.precipitation))

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

    def __add__(self, other):
        assert isinstance(other, RadolanSingleStat)
        raise NotImplementedError


