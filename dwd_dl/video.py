"""This module contains the relevant tools to make a video from a model and a datamodule

"""
import warnings
import os
import datetime as dt

import matplotlib as mpl
# mpl.use('Agg')
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
                 video_mode,
                 frame_rate: float = 30.):
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        if video_mode not in ('pred', 'true', 'both'):
            raise ValueError(f"{video_mode} is not a valid video_mode for {type(self)}")
        self.video_mode = video_mode
        self.frame_rate = frame_rate
        self.timestamp_string = self.model.timestamp_string
        if self.timestamp_string is None:
            warnings.warn(f"timestamp string is {self.timestamp_string}")
        self.dir_path = os.path.join(cfg.CFG.RADOLAN_ROOT, 'Videos', self.timestamp_string)

    def produce(self):
        self._make_dirs()
        sequences = []
        if self.video_mode in ('pred', 'both'):
            predictions_array = self._make_prediction_array()
            prediction_sequence = VideoRadarSequence(predictions_array, cfg.CFG.video_ranges, 'pred')
            sequences.append(prediction_sequence)
        if self.video_mode in ('true', 'both'):
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
        prediction_groups = [np.squeeze(np.argmax(batch, axis=1)) for batch in prediction]
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
            default=10,
            help="The video framerate. (default: 10)"
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="The path to the saved model."
        )
        parser.add_argument(
            "--video_mode",
            type=str,
            default='both',
            help="The video_mode of the video. 'pred', 'true' or 'both'. (default: 'both')"
        )
        return parent_parser

    def video_name(self, mode=None):
        if mode is None:
            mode = self.video_mode
        if mode not in ('pred', 'true', 'both'):
            raise ValueError(f"{mode} is not a valid video_mode for {type(self)}")

        base_name = f"{self.timestamp_string}_{mode}"
        if not os.path.isdir(self.dir_path):
            print(f"Directory {self.dir_path} does not exist.")
            return base_name + '.mp4'
        else:
            def file_name(v): return base_name + f"_v{v}" + ".mp4"

            for v in range(1, 100):
                if not os.path.isfile(os.path.join(self.dir_path, file_name(v))):
                    return file_name(v)
            raise OverflowError("Too many versions. Clear them up!")

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
        return self.video_name(mode=self.video_mode).split('.mp4')[0] + '.yml'

    def _save_mp4(self, sequences, datetimes):

        sequences_dict = dict()
        for sequence in sequences:
            sequences_dict[sequence.name] = sequence

        delta_t = 1. / self.frame_rate

        fig = plt.figure()

        ax = dict()
        number_of_subplots = len(sequences_dict)
        cols = number_of_subplots
        rows = 1
        for n, sequence_name in enumerate(sequences_dict):
            subplot_code = rows * 100 + cols * 10 + 1 + n
            ax[sequence_name] = fig.add_subplot(subplot_code, aspect='equal', autoscale_on=True)
            time_text = {
                sequence: ax[sequence].text(0.02, 1.15, '', transform=ax[sequence].transAxes) for sequence in ax
            }

        def init():
            """initialize animation"""
            # line.set_data([], [])
            out = [text.set_text('') for text in time_text.values()]
            return out

        def animate(i):
            """perform animation step"""
            array_plot = []
            text = []
            for seq in sequences_dict:
                if i > 0:
                    sequences_dict[seq].step()

                im = ax[seq].imshow(sequences_dict[seq].state, vmin=0, vmax=4)
                array_plot.append(im)
                text.append(
                    time_text[seq].set_text(
                        f"{seq}\ntime elapsed = {sequences_dict[seq].time_elapsed} \ndatetime of T-6h = {datetimes[i]}"
                    )
                )
            fig.canvas.draw()
            return array_plot + text

        animate(0)

        interval = 1000 * delta_t

        ani = animation.FuncAnimation(fig, animate, frames=len(list(sequences_dict.values())[0]),
                                      interval=interval, init_func=init)

        path_to_mp4 = os.path.join(self.dir_path, self.video_name())
        ani.save(path_to_mp4, fps=self.frame_rate)  # , extra_args=['-vcodec', 'libx264'])

    def _make_dirs(self):
        os.makedirs(self.dir_path, exist_ok=True)
