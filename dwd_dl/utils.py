"""various utilities

"""

import datetime as dt
import os
from typing import List

import torch
from torch.utils.data import DataLoader
import numpy as np

import dwd_dl.cfg as cfg
import dwd_dl.unet as unet


def init_safety(func):  # Nice decorator in case it is needed.
    def wrapper(*args, **kwargs):
        if cfg.CFG is None and not isinstance(cfg.CFG, cfg.Config):
            raise UserWarning("Before using {} please run dwd_dl.cfg.initialize".format(func))
        return func(*args, **kwargs)
    return wrapper


def create_checkpoint_path(experiment_timestamp_str):
    checkpoint_name = create_checkpoint_name(experiment_timestamp_str)
    checkpoint_dir = os.path.join(cfg.CFG.RADOLAN_ROOT, 'Models', experiment_timestamp_str)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    return checkpoint_path


def create_checkpoint_name(experiment_timestamp_str):
    """Change prefix if you want to change the checkpoint name.

    """
    prefix = ''
    return prefix + f"{experiment_timestamp_str}.ckpt"


def unet_saver(trained_network, path=None, fname=None, timestamp=None):
    if path is None:
        path = input('Enter a path for saving the trained model. ')

    if fname is None:
        fname = network_name_timestamp(timestamp=timestamp)

    path_to_file = os.path.join(os.path.abspath(path), fname)
    basedir = os.path.dirname(path_to_file)
    os.makedirs(basedir, exist_ok=True)
    torch.save(trained_network.state_dict(), path_to_file)

    return path_to_file


def network_name_timestamp(timestamp=None):
    name_template = 'UNet_run_{}.pt'
    if timestamp is not None:
        name = name_template.format(timestamp)
    else:
        name = name_template.format(dt.datetime.now())

    return name


def network_loader(path_to_saved_model, network_class, eval_or_train, *args, **kwargs):
    model = network_class(*args, **kwargs)
    model.load_state_dict(torch.load(path_to_saved_model))
    if eval_or_train == 'eval':
        model.eval()
    elif eval_or_train == 'train':
        model.train()

    return model


class ModelEvaluator:
    def __init__(
            self,
            h5file_handle,
            path_to_saved_model,
            network_class,
            date_ranges_path,
            eval_or_train='eval',
            device='cuda:0'
    ):
        import dwd_dl.dataset as ds
        self._phases = ['train', 'valid']
        self.model = network_loader(path_to_saved_model, network_class, eval_or_train,
                                    permute_output=False, softmax_output=False)
        if torch.cuda.is_available():
            self._device = device
        else:
            self._device = 'cpu'
        self._true_dataset = ds.RadolanDataset(h5file_handle=h5file_handle, date_ranges_path=date_ranges_path,
                                               image_size=256, in_channels=6, out_channels=1)

        def worker_init(worker_id):
            np.random.seed(42 + worker_id)

        self._train_dataset = ds.RadolanSubset(dataset=self._true_dataset, subset='train')
        self._valid_dataset = ds.RadolanSubset(dataset=self._true_dataset, subset='valid')
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=1,
            num_workers=8,
            worker_init_fn=worker_init
        )
        self._valid_loader = DataLoader(
            self._valid_dataset,
            batch_size=1,
            num_workers=8,
            worker_init_fn=worker_init
        )

        self._dataset = {'train': self._train_dataset, 'valid': self._valid_dataset}

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        assert device in ['cpu', 'cuda:0']  # TODO: write better implementation
        self._device = device
        print('Warning, this way of setting device may be not complete.')

    def which_dataset(self, timestamp):
        valid = self._valid_dataset.has_timestamp(timestamp)
        train = self._train_dataset.has_timestamp(timestamp)
        if not valid and not train:
            raise ValueError('Current timestamp {} is not present, neither '
                             'in validation nor in training dataset.'.format(timestamp))
        elif valid and train:
            raise ValueError('Current timestamp {} is present in both validation and training dataset.'
                             'This error should never occur. Check code.'.format(timestamp))
        elif valid:
            return 'valid'
        else:
            return 'train'

    def on_timestamp(self, timestamp):
        phase = self.which_dataset(timestamp)
        return self._evaluate(phase, timestamp)

    def _evaluate(self, phase, timestamp, to_numpy=True):
        assert phase in ['valid', 'train']
        data = self._dataset[phase].from_timestamp(timestamp)
        model = self.model
        device = self.device
        model.to(device)
        x, y_true = data
        x, y_true = torch.unsqueeze(x, 0), torch.unsqueeze(y_true, 0)
        x, y_true = x.to(device), y_true.to(device)
        y = model(x)

        x, y, y_true = x[:, ::4], torch.squeeze(torch.topk(y, 1, dim=2).indices, dim=0), y_true[:, ::4]
        x, y, y_true = x.cpu().detach().numpy(), y.cpu().detach().numpy(), y_true.cpu().detach().numpy()

        return x, y, y_true

    def all_timestamps(self):
        return self._true_dataset.sorted_sequence

    @property
    def legal_timestamps(self):
        return sorted(self._valid_dataset.timestamps + self._train_dataset.timestamps)

    def all(self, timestamp_list=None):
        if timestamp_list is None:
            timestamp_list = self.all_timestamps()

        for timestamp in timestamp_list:
            try:
                yield timestamp, self.on_timestamp(timestamp)
            except ValueError:
                continue

    def all_split(self):
        out = {}
        for phase in self._phases:
            out[phase] = self._dataset[phase].timestamps
        return out


def get_images_arrays(dates_list: List[dt.datetime], path_to_saved_model, eval_or_train='eval', batch_size=None):
    import dwd_dl.dataset as ds

    with ds.h5dataset_context_wrapper(cfg.CFG.date_ranges, mode=cfg.CFG.MODE, classes=cfg.CFG.CLASSES) as f:
        evaluator = ModelEvaluator(
            h5file_handle=f,
            path_to_saved_model=path_to_saved_model,
            network_class=unet.UNet,
            date_ranges_path=cfg.CFG.DATE_RANGES_FILE_PATH,
            eval_or_train=eval_or_train
        )
        images_arrays = []
        with torch.no_grad():
            legal_timestamps = evaluator.legal_timestamps
            for date in dates_list:
                timestamp = date.strftime(cfg.CFG.TIMESTAMP_DATE_FORMAT)
                if date.strftime(cfg.CFG.TIMESTAMP_DATE_FORMAT) in legal_timestamps:
                    images_arrays.append(evaluator.on_timestamp(timestamp))
    return images_arrays


def to_class_index(tensor: torch.tensor, dtype=torch.long, device=torch.device('cuda:0'), numpy_or_torch='n'):
    """This function converts a tensor of shape (B,N,C,H,W) to (B,N,H,W) collapsing C into N as class index.
    It should only contain ones and zeros. The output are indices as torch.long.

    works also with 3 and 4 dimensions

    Parameters
    ----------
    device
    numpy_or_torch
    dtype
    tensor

    Returns
    -------

    """
    ndim = tensor.ndim
    assert ndim in (3, 4, 5)
    assert numpy_or_torch in ('n', 't')
    axes_to_add = tuple(i for i in range(0, 5 - ndim))
    np_array = np.expand_dims(tensor, axes_to_add)

    category_indices = np.zeros((*np_array.shape[:2], *np_array.shape[-2:]))
    for category_number in range(np_array.shape[2]):
        category_indices += category_number * np_array[:, :, category_number, ...]

    category_indices = np.squeeze(category_indices, axis=axes_to_add)
    if numpy_or_torch == 't':
        category_indices = torch.from_numpy(category_indices)
        category_indices.to(device=device)
        category_indices.to(dtype=dtype)

    return category_indices


@init_safety
def visualize(path_to_saved_model, *args, eval_or_train='eval', **kwargs):
    import dwd_dl.dataset as ds
    import dwd_dl.img as img
    with ds.h5dataset_context_wrapper(*args, **kwargs) as f:
        evaluator = ModelEvaluator(
            h5file_handle=f,
            path_to_saved_model=path_to_saved_model,
            network_class=unet.UNet,
            date_ranges_path=cfg.CFG.DATE_RANGES_FILE_PATH,
            eval_or_train=eval_or_train
        )
        with torch.no_grad():
            img.visualizer(evaluator)


def incremental_std(std, mean, sequence_n, elements_in_image, std_at_timestamp, mean_at_timestamp):
    return np.sqrt(((sequence_n * elements_in_image - 1) * (std ** 2) + (elements_in_image - 1)*(std_at_timestamp**2))/(
            (sequence_n + 1)*elements_in_image - 1)) + ((sequence_n * elements_in_image * elements_in_image) * (
                (mean - mean_at_timestamp) ** 2) / (((sequence_n+1) * elements_in_image)*(
                    (sequence_n+1) * elements_in_image - 1))
            )


def incremental_mean(mean, sequence_n, elements_in_image, mean_at_timestamp):
    return (sequence_n * elements_in_image * mean + (mean_at_timestamp * elements_in_image)) / (
            (sequence_n + 1) * elements_in_image
    )


def square_select(time_stamp, height=None, width=None, plot=False):
    import dwd_dl.img as img
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

    if not height:
        height = cfg.CFG.HEIGHT
    if not width:
        width = cfg.CFG.WIDTH

    out = img.selection(time_stamp, plot=False).RW[
          cfg.CFG.NW_CORNER_INDICES[0] - height:cfg.CFG.NW_CORNER_INDICES[0],
          cfg.CFG.NW_CORNER_INDICES[1]:cfg.CFG.NW_CORNER_INDICES[1] + width
          ]
    if plot:
        out.plot()

    return out.copy()


def cut_square(array, height, width, indices_up_left):
    return array[indices_up_left[0] - height:indices_up_left[0], indices_up_left[1]:indices_up_left[1] + width]


def year_month_tuple_list(start_date, end_date):
    start_year, start_month = start_date.year, start_date.month
    year = start_year
    month = start_month
    result = []
    while month + 12 * (year - end_date.year) <= end_date.month:  # this is a linear inequality
        result.append((year, month))
        year, month = next_year_month(year, month)
    return result


def next_year_month(year, month):
    months_per_year = 12
    month += 1
    year += month // (months_per_year + 1)
    month -= months_per_year * (month // (months_per_year + 1))
    return year, month


def ym_tuples(date_ranges):
    year_month_tuples = []
    for date_range in date_ranges:
        year_month_tuples.extend(
            ym for ym in year_month_tuple_list(date_range.start, date_range.end) if ym not in year_month_tuples
        )
    return year_month_tuples


def normalized_time_of_day_from_string(timestamp_string, min_=-1, max_=1):
    minute = int(timestamp_string[-2:])
    hour = int(timestamp_string[-4:-3])
    hour_minute = hour + (minute / 60)
    hours_in_day = 24
    std = max_ - min_
    mean = max_ - std / 2
    return (hour_minute / hours_in_day) * std - mean
