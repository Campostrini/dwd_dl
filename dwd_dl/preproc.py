"""This file contains useful preprocessing tools

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

    """

    out = img.selection(time_stamp, plot=False).RW[nw_corner_indexes[0] - height:nw_corner_indexes[0],
                                                   nw_corner_indexes[1]:nw_corner_indexes[1] + width]
    if plot:
        out.plot()

    return out.copy()
