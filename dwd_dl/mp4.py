"""
General Numerical Solver for the 1D Time-Dependent Schrodinger's equation.

adapted from code at http://matplotlib.sourceforge.net/examples/animation/double_pendulum_animated.py

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import datetime as dt
from typing import Union

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import dwd_dl.cfg as cfg
from .utils import get_images_arrays, to_class_index


class RadarSequence:
    def __init__(self, start_date: dt.datetime, end_date: dt.datetime, batch_size: int, path_to_saved_model: str):
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self._dates_list = [date for date in cfg.daterange(start_date, end_date, include_end=True)]
        self._images_arrays = get_images_arrays(self._dates_list, path_to_saved_model, batch_size=batch_size)
        self._images_arrays = [
            np.concatenate([np.squeeze(im_item, axis=0) for im_item in im]) for im in self._images_arrays
        ]
        self.state = self._images_arrays[0]
        self._step_index = 0
        self.time_elapsed = dt.timedelta(0)
        self.current_time = start_date

    def step(self):
        self.time_elapsed += dt.timedelta(hours=1)

        self._step_index += 1
        self.current_time = self._dates_list[self._step_index]
        self.state = self._images_arrays[self._step_index]
        print(self.current_time)


def save_mp4(start_date: dt.datetime, end_date: dt.datetime, path_to_saved_model, path_to_mp4,
             batch_size: int = 5, image_index: Union[int, None] = None):

    if image_index is None:
        image_index = 6
    assert image_index in range(8)
    radar_sequence = RadarSequence(start_date=start_date, end_date=end_date, batch_size=batch_size,
                                   path_to_saved_model=path_to_saved_model)
    delta_t = 1./30

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=True)

# # set up figure and animation
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
#                      xlim=(-2, 2), ylim=(-2, 2))
# ax.grid()
#
# line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        """initialize animation"""
        # line.set_data([], [])
        time_text.set_text('')
        return time_text

    def animate(i):
        """perform animation step"""
        radar_sequence.step()

        array_plot = ax.imshow(np.squeeze(radar_sequence.state[image_index]))
        time_text.set_text(f'time = {radar_sequence.time_elapsed}')
        fig.canvas.draw()
        return array_plot, time_text

    animate(0)
    interval = 1000 * delta_t

    ani = animation.FuncAnimation(fig, animate, frames=150,
                                  interval=interval, init_func=init)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    ani.save(path_to_mp4, fps=30, extra_args=['-vcodec', 'libx264'])

