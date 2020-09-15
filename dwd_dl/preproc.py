"""This file contains useful preprocessing tools.

Some preprocessing tools for preparing the data for the U-Net phase.

"""

from . import config
from . import img

import numpy as np
import datetime as dt

NW_CORNER_COORD = np.array([8.6, 53.6])

height = width = 256

nw_corner_indexes = config.coords_finder(*NW_CORNER_COORD, distances_output=False)


# img.selection(time_stamp=ts, plot=False)


def square_select(time_stamp, height=256, width=256, plot=False):
    """Returns the square selection of an area with NW_CORNER set

    Parameters
    ----------
    time_stamp : datetime.datetime
        A timestamp for which the DWD Radolan data is available.
    height : int, optional
        The number of pixels of the height of the selection. (Defaults to 256, i.e. 256 km. Don't change unless there
        is a valid reason.)
    width : int, optional
        The number of pixels of th width of the selection. (Defaults to 256, i.e. 256 km. Don't change unless there
        is a valid reason.)
    plot : bool, optional
        Whether the result should be plotted or not.

    Returns
    -------
    xarray.core.dataarray.DataArray
        A copy of the selection.

    """

    out = img.selection(time_stamp, plot=False).RW[nw_corner_indexes[0] - height:nw_corner_indexes[0],
                                                   nw_corner_indexes[1]:nw_corner_indexes[1] + width]
    if plot:
        out.plot()

    return out.copy()


def normalize(array, mean, std):
    return (array - mean) / std


def unnormalize(array, mean, std):
    return array * std + mean
