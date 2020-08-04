"""Image manipulation library for visualizing the DWD data.

Makes use of wradlib.

"""
import datetime
import os
import warnings
import matplotlib.pyplot as pl

import wradlib as wrl

from dwd_dl import config


def selection(
        time_stamp,
        *,
        center_coords=None,
        xy_widths=None,
        bl_coords=None,
        tr_coords=None,
        plot=True,
        verbose=False
):
    """Selection tool for DWD Radolan data.

    Uses wradlib and outputs an xarray.DataSet. Plot functionality included.

    Parameters
    ----------
    verbose
    time_stamp : datetime.datetime
        Datetime valid timestamp for which DWD Radolan data exists.
    center_coords : tuple or list of length 2
        Prefer tuple. The coordinates of the center of a selection. In polar stereographic.
    xy_widths : tuple or list of length 2
        Prefer tuple. The widths of the selection. In polar stereographic.
    bl_coords : tuple or list of length 2
        Prefer tuple. The bottom left point coordinates of the selection. In polar stereographic.
    tr_coords : tuple or list of length 2
        Prefer tuple. The top right point coordinates of the selection. In polar stereographic.
    plot : bool
        Whether a plot should be produced or not.
    verbose : bool
        Suppresses or enables printing statements.

    Returns
    -------
    xarray.DataSet
        An xarray.DataSet copy of the data.

    """
    # Initial checks for time_stamp
    while True:
        try:
            assert config.START_DATE <= time_stamp <= config.END_DATE, 'Wrong input. Time is not ' \
                                                                       'in the expected interval.'
        except AssertionError:
            print('No selection was produced. Retry with a valid date.')
            return None
        except AttributeError:
            print('Configuration was not run yet. Running it now.')
            config.config_initializer()
            print('Configuration run.')
        else:
            break

    # Initial checks for coordinates.
    if center_coords is not None and xy_widths is not None and (bl_coords, tr_coords) == (None, None):
        # assert isinstance(center_coords, tuple), 'center_coords should be a tuple.'
        # assert isinstance(xy_widths, tuple), 'xy_widths should be a tuple.'
        # assert len(center_coords) == 2, 'The length of center_coords should be 2.'
        # assert len(xy_widths) == 2, 'The length of xy_widths should be 2.'
        # assert 0 <= center_coords[0] <= 900 and 0 <= center_coords[1] <= 900
        # # TODO: Add other checks. Really need all those?
        # assert 0 <= xy_widths[0] <= 900 and 0 <= xy_widths[1] <= 900
        x_slice = slice(center_coords[0]-0.5*xy_widths[0], center_coords[0]+0.5*xy_widths[0])
        y_slice = slice(center_coords[1]-0.5*xy_widths[1], center_coords[1]+0.5*xy_widths[1])
    elif bl_coords is not None and tr_coords is not None and (center_coords, xy_widths) == (None, None):
        x_slice = slice(bl_coords[0], tr_coords[0])
        y_slice = slice(bl_coords[1], tr_coords[1])
    else:
        if verbose:
            print('Will not slice.')
        x_slice, y_slice = None, None

    # Open the data
    file_path = os.path.join(config.RADOLAN_PATH, config.binary_file_name(time_stamp))
    rw_filename = wrl.util.get_wradlib_data_file(file_path)
    ds, rwattrs = wrl.io.read_radolan_composite(rw_filename, loaddata='xarray')

    if (x_slice, y_slice) != (None, None):
        selection = ds.sel(x=x_slice, y=y_slice)
    else:
        selection = ds

    # Visualize

    warnings.filterwarnings('ignore')
    try:
        get_ipython().magic("matplotlib inline")
    except:
        pl.ion()

    if plot:
        selection.RW.plot()

    # Return xarray.DataSet copy of the data.

    return selection.copy()



# TODO: Compute stats