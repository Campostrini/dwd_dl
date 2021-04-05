"""This module contains the relevant tools to make a video from a model and a datamodule

"""
import warnings
import os
import datetime as dt

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytorch_lightning import Trainer, LightningModule
import numpy as np
import yaml

import dwd_dl.cfg as cfg
from dwd_dl.data_module import VideoDataModule


class VideoRadarSequence:
    def __init__(self, sequence_array: np.ndarray, time_range, name):
        assert name in ('pred', 'true')
        self.sequence_array = sequence_array
        self.time_range = time_range
        self._dates_list = []
        for date_range in self.time_range:
            self._dates_list.extend([date for date in date_range.date_range()])
        self.state = self.sequence_array[0]
        self._step_index = 0
        self.time_elapsed = dt.timedelta(0)
        self.current_time = self._dates_list[0]
        self.name = name

    def step(self):
        self.time_elapsed += dt.timedelta(hours=1)
        self._step_index += 1
        self.current_time = self._dates_list[self._step_index]
        self.state = self.sequence_array[self._step_index]
        print(self.current_time)

    def __len__(self):
        return len(self.sequence_array)


class VideoProducer:
    def __init__(self,
                 trainer: Trainer,
                 model: LightningModule,
                 datamodule: VideoDataModule,
                 mode,
                 frame_rate: float = 30.):
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        if mode not in ('pred', 'true', 'both'):
            raise ValueError(f"{mode} is not a valid mode for {type(self)}")
        self.mode = mode
        self.frame_rate = frame_rate
        self.timestamp_string = self.model.timestamp_string
        if self.timestamp_string is None:
            warnings.warn(f"timestamp string is {self.timestamp_string}")
        self.dir_path = os.path.join(cfg.CFG.RADOLAN_ROOT, 'Videos', self.timestamp_string)

    def produce(self):
        self._make_dirs()
        sequences = []
        if self.mode in ('pred', 'both'):
            predictions_array = self._make_prediction_array()
            prediction_sequence = VideoRadarSequence(predictions_array, cfg.CFG.VIDEO_RANGES, 'pred')
            sequences.append(prediction_sequence)
        if self.mode in ('true', 'both'):
            print("making True array")
            true_array = self._make_true_array()
            print("initialising sequence")
            true_sequence = VideoRadarSequence(true_array, cfg.CFG.video_ranges, 'true')
            sequences.append(true_sequence)
        datetimes = self._legal_datetimes()
        self._yaml_dump()
        self._save_mp4(sequences, datetimes)

    def predict(self):
        prediction = self.trainer.predict(model=self.model, datamodule=self.datamodule)
        return prediction

    def _make_prediction_array(self):
        prediction = self.predict()
        prediction_groups = (np.argmax(batch, axis=1) for batch in prediction)
        return np.concatenate(prediction_groups, axis=0)

    def _make_true_array(self):
        if not self.datamodule.has_setup_fit:
            self.datamodule.setup()
        true_groups = [y_true[:, 0, ...].cpu().numpy() for x, y_true in self.datamodule.predict_dataloader()]
        return np.concatenate(true_groups, axis=0)

    @staticmethod
    def add_video_specific_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Video")
        parser.add_argument(
            "--video",
            type=bool,
            default=True,
            help="Whether a video should be produced. (default: true)"
        )
        parser.add_argument(
            "--frame_rate",
            type=float,
            default=30,
            help="The video framerate. (default: 30)"
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="The path to the saved model."
        )
        return parent_parser

    def video_name(self, mode=None):
        if mode is None:
            mode = self.mode
        if mode not in ('pred', 'true', 'both'):
            raise ValueError(f"{mode} is not a valid mode for {type(self)}")

        base_name = f"{self.timestamp_string}_{mode}"
        if not os.path.isdir(self.dir_path):
            print(f"Directory {self.dir_path} does not exist.")
            return base_name + '.mp4'
        else:
            version = 1
            def file_name(v): return base_name + f"_v{v}" + ".mp4"
            while os.path.isfile(os.path.join(self.dir_path, file_name(version))):
                version += 1
                if version > 100:
                    raise OverflowError("Too many versions. Clear them up!")
            return file_name(version)

    def _legal_datetimes(self):
        return self.datamodule.legal_datetimes()

    def _simplify_legal_datetimes(self):
        datetimes = sorted(self._legal_datetimes())
        simplified_datetimes = []
        start = end = datetimes[0]
        for datetime in datetimes[1:]:
            if datetime > end + dt.timedelta(hours=2):
                simplified_datetimes.append([start, end])
                start = datetime
                end = datetime
            else:
                end = datetime
        simplified_datetimes.append([start, end])
        return simplified_datetimes

    def _yaml_dump(self):
        simplified_datetimes = self._simplify_legal_datetimes()
        yaml_name = self._yaml_name()
        with open(os.path.join(self.dir_path, yaml_name), mode='w') as f:
            yaml.safe_dump(simplified_datetimes, f, default_flow_style=False)

    def _yaml_name(self):
        return self.video_name(mode=self.mode).split('.mp4')[0] + '.yml'


    def _save_mp4(self, sequences, datetimes):

        sequences_dict = dict()
        for sequence in sequences:
            sequences_dict[sequence.name] = sequence

        delta_t = 1. / self.frame_rate

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=True)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            """initialize animation"""
            # line.set_data([], [])
            time_text.set_text('')
            return time_text

        def animate(i):
            """perform animation step"""
            if i > 0:
                sequences_dict['true'].step()
            array_plot = ax.imshow(sequences_dict['true'].state)
            time_text.set_text(f"time = {sequences_dict['true'].time_elapsed} \n datetime = {datetimes[i]}")
            fig.canvas.draw()
            return array_plot, time_text

        animate(0)
        interval = 1000 * delta_t

        ani = animation.FuncAnimation(fig, animate, frames=len(sequences_dict['true']),
                                      interval=interval, init_func=init)

        path_to_mp4 = os.path.join(self.dir_path, self.video_name('true'))
        ani.save(path_to_mp4, fps=self.frame_rate)  # , extra_args=['-vcodec', 'libx264'])

    def _make_dirs(self):
        os.makedirs(self.dir_path, exist_ok=True)
