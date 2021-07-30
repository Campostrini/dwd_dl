# Radolan Stat Abstract Class
from abc import ABC, abstractmethod
import datetime as dt

import numpy as np

from . import cfg


class RadolanStatAbstractClass(ABC):
# Takes numpy histogram as input, with limits

# Hist property
    @abstractmethod
    def hist(self):
        pass

# Bins property
    @abstractmethod
    def bins(self):
        pass

# Whisker preparation for plot
    @abstractmethod
    def box_plot_dict(self):
        pass


class RadolanSingleStat(RadolanStatAbstractClass):
    def __init__(self, histogram_frequencies, histogram_bins, timestamp, name):
        if not len(histogram_bins) == len(histogram_frequencies) + 1:
            raise ValueError("histogram_frequencies must be of len len(histogram_bins) - 1 . \n"
                             f"Got {len(histogram_frequencies)} and {len(histogram_bins)} instead.")
        self._timestamp = timestamp
        self._time = dt.datetime.strptime(timestamp, cfg.CFG.TIMESTAMP_DATE_FORMAT)
        self._histogram_bins = histogram_bins
        self._histogram_frequencies = histogram_frequencies
        self._name = name

    def hist(self):
        return self._histogram_frequencies, self._histogram_bins

    def bins(self):
        return self._histogram_bins

    def box_plot_dict(self):
        q1, median, q3 = quantiles_calc_from_hist(self.hist(), [0.25, 0.5, 0.75])
        return {
            'med': median,
            'q1': q1,
            'q3': q3,
            'whislo': None,
            'whishi': None,
            'fliers': None,
            'label': None,
        }

    def __add__(self, other):
        assert isinstance(other, (RadolanSingleStat, RadolanMultiStat))


class RadolanMultiStat(RadolanStatAbstractClass):

    # should return whether a timestamp or a datetime is contained in this multistat
    def __contains__(self, item):
        pass

    def percentage_with_rain(self):
        pass


def quantile_calc_from_hist(histogram, quantile):
    assert 0 < quantile < 1
    frequencies, bins = histogram
    sum_ = sum(frequencies)
    frequency_of_quantile = sum_/(1/quantile)
    partial = 0
    quantile_value = None
    for i, freq in enumerate(frequencies):
        partial += freq
        if partial > frequency_of_quantile:
            quantile_index = i
            width = bins[quantile_index + 1] - bins[quantile_index]
            lower_limit = bins[quantile_index]
            cumulative_up_to_group = partial - freq
            # quantile value estimate (linear interpolation)
            quantile_value = lower_limit + ((frequency_of_quantile - cumulative_up_to_group) / freq) * width
            assert bins[quantile_index] < quantile_value < bins[quantile_index + 1]
            break
        elif partial == frequency_of_quantile:
            quantile_index = i
            quantile_value = bins[quantile_index + 1]
            break
    if quantile_value is None:
        raise ValueError(f"Ooops, no quantile {quantile} was computed. Something went wrong")


def quantiles_calc_from_hist(histogram, list_of_quantiles):
    all(0 < quantile < 1 for quantile in list_of_quantiles)
    return [quantile_calc_from_hist(histogram, quantile) for quantile in list_of_quantiles]

