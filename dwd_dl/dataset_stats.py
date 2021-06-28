from dwd_dl import cfg
import dwd_dl.dataset as ds

import pandas as pd
from xarray import open_mfdataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cfg.initialize(skip_download=True)
    if ds.check_h5_missing_or_corrupt(cfg.CFG.date_ranges, classes=cfg.CFG.CLASSES, mode=cfg.CFG.MODE):
        cfg.initialize()
        ds.create_h5(mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES)

    radolan_dataset = ds.RadolanDataset()

    df = pd.DataFrame.from_dict(radolan_dataset.classes_frequency, orient='index')
    df.index = pd.to_datetime(df.index, format=cfg.CFG.TIMESTAMP_DATE_FORMAT)

    test_set_ts = [pd.Series(series.date_range()) for series in cfg.CFG.test_set_ranges]
    test_set_ts = test_set_ts[0].append(test_set_ts[1:])

    test_set_df = df[df.index.isin(test_set_ts)]
    non_test_set_df = df[~df.index.isin(test_set_ts)]

    plt.figure()

    test_set_df.groupby(test_set_df.index.date).sum().plot()
    non_test_set_df.groupby(non_test_set_df.index.date).sum().plot()

    classes_percentages_test_set = test_set_df.sum() * 100 / test_set_df.to_numpy().sum()
    print("Classes percentages in test set: \n{}".format(classes_percentages_test_set))

    classes_percentages_non_test_set = non_test_set_df.sum() * 100 / non_test_set_df.to_numpy().sum()
    print("Classes percentages in non test set: \n{}".format(classes_percentages_non_test_set))

    # PANDAS
    grid = ['/shared1/RadolanData/Radolan/H5/v0.0.5-r-2015010.h5',
            '/shared1/RadolanData/Radolan/H5/v0.0.5-r-2016010.h5']
    ds = open_mfdataset(grid)
    da = ds.to_array()
    da = da.rename({"variable": "time", "phony_dim_0": "channels", "phony_dim_1": "y", "phony_dim_2": "x"})
    da = da.assign_coords(time=pd.to_datetime(da.coords["time"].values, format=cfg.CFG.TIMESTAMP_DATE_FORMAT))
    da = da[:, 0, ...]
    df = da.to_dataframe()
    df = df.droplevel(-1)
    df = df.droplevel(-1)
    df.boxplot(by=pd.Grouper(freq="M"), rot=90)
    plt.show()
