"""Image manipulation library

"""
import datetime

import wradlib as wrl

from dwd_dl import config


def selection(time_stamp, *, center_coords=None, xy_widths=None, bl_coords=None, tr_coords=None, plot=False):

    # Initial checks for time_stamp
    while True:
        try:
            assert config.START_DATE < time_stamp < config.END_DATE, 'Wrong input. Time is not ' \
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
    if center_coords is not None and xy_widths is not None and (bl_coords, tr_coords) is (None, None):
        # assert isinstance(center_coords, tuple), 'center_coords should be a tuple.'
        # assert isinstance(xy_widths, tuple), 'xy_widths should be a tuple.'
        # assert len(center_coords) == 2, 'The length of center_coords should be 2.'
        # assert len(xy_widths) == 2, 'The length of xy_widths should be 2.'
        # assert 0 <= center_coords[0] <= 900 and 0 <= center_coords[1] <= 900
        # # TODO: Add other checks. Really need all those?
        # assert 0 <= xy_widths[0] <= 900 and 0 <= xy_widths[1] <= 900
        x_slice = slice(center_coords[0]-0.5*xy_widths[0], center_coords[0]+0.5*xy_widths[0])
        y_slice = slice(center_coords[1]-0.5*xy_widths[1], center_coords[1]+0.5*xy_widths[1])
    elif bl_coords is not None and tr_coords is not None and (center_coords, xy_widths) is (None, None):
        x_slice = slice(bl_coords[0], tr_coords[0])
        y_slice = slice(bl_coords[1], tr_coords[1])
    else:
        print('Wrong input. Will not slice.')
        x_slice, y_slice = None, None

    # Open the data


    # Return np copy of the data if copy is True
    # Visualize

def simple_plot(time_stamp):
    return None

    # TODO: Compute stats