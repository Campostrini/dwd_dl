import datetime as dt

import pandas as pd

import cfg
from dataset import RadolanDataset


def get_total_pre_time_series_with_timestamps(radolan_dataset: RadolanDataset) -> pd.Series:
    time_series_dict = dict()
    for i in range(len(radolan_dataset)):
        seq, tru = radolan_dataset.indices_tuple[i]
        timestamp = dt.datetime.strptime(radolan_dataset.sorted_sequence[seq[0]], cfg.CFG.TIMESTAMP_DATE_FORMAT)
        time_series_dict[timestamp] = 0
        for indices in (seq, tru):
            for t in indices:
                time_series_dict[timestamp] += radolan_dataset._tot_pre[radolan_dataset.sorted_sequence[t]]
    return pd.Series(time_series_dict)


def get_nonzero_timestamps(radolan_dataset: RadolanDataset):
    precipitation_series = get_total_pre_time_series_with_timestamps(radolan_dataset)
    return precipitation_series[precipitation_series > 0]


def get_nonzero_intervals(radolan_dataset: RadolanDataset):
    precipitation_series = get_total_pre_time_series_with_timestamps(radolan_dataset)
    open_ = False
    start = end = None
    interval_list = []
    for i, (index, value) in enumerate(precipitation_series.iteritems()):
        if value == 0 and not open_:
            continue
        elif value > 0 and not open_:
            start = index
            end = index
            open_ = True
            continue
        elif value > 0 and open_:
            if not index - end == dt.timedelta(hours=1):
                interval = pd.Interval(start, end, closed='both')
                interval_list.append(interval)
                open_ = False
                continue
            end = index
            continue
        elif value == 0 and open_:
            interval = pd.Interval(start, end, closed='both')
            open_ = False
            interval_list.append(interval)
            continue
    return pd.Series(interval_list)

