"""Module for checking requirements and downloading the radar data.

"""
import gzip
import hashlib
import inspect
import os
import shutil
import urllib.request
import requests
import itertools
import warnings
import datetime as dt
import tempfile
import sys
import tarfile
import numpy as np
import wradlib as wrl
from osgeo import osr
from tqdm import tqdm
from packaging import version
from typing import List

import dwd_dl as dl
import dwd_dl.utils as utils
import dwd_dl.yaml_utils as yu
import dwd_dl.dataset as ds

CFG = None
CONFIG_WAS_RUN = False


# TODO: Add download manager for current year


class RadolanConfigFileContent:
    def __init__(
            self,
            BASE_URL: str,
            RADOLAN_ROOT: str,
            RANGES_DATE_FORMAT: str,
            TIMESTAMP_DATE_FORMAT: str,
            DATES: dict,
            NW_CORNER_LON_LAT: list,
            HEIGHT: int,
            WIDTH: int,
            H5_VERSION: str,
            MODE: str,
            CLASSES: dict,
            VSC: bool,
            VIDEO: dict,
    ):
        self._BASE_URL = BASE_URL
        self._RADOLAN_ROOT = RADOLAN_ROOT
        self._RANGES_DATE_FORMAT = RANGES_DATE_FORMAT
        self._TIMESTAMP_DATE_FORMAT = TIMESTAMP_DATE_FORMAT
        self._MIN_START_DATE = DATES['MIN_START_DATE']
        self._MAX_END_DATE = DATES['MAX_END_DATE']
        self._NW_CORNER_LON_LAT = NW_CORNER_LON_LAT
        self._HEIGHT = HEIGHT
        self._WIDTH = WIDTH
        self._H5_VERSION = H5_VERSION
        self._MODE = MODE
        self._CLASSES = CLASSES
        self._VSC = VSC
        self._VIDEO_START = VIDEO['START']
        self._VIDEO_END = VIDEO['END']

    @property
    def BASE_URL(self):
        return self._BASE_URL

    @property
    def RADOLAN_ROOT(self):
        return self._RADOLAN_ROOT

    @property
    def RANGES_DATE_FORMAT(self):
        return self._RANGES_DATE_FORMAT

    @property
    def TIMESTAMP_DATE_FORMAT(self):
        return self._TIMESTAMP_DATE_FORMAT

    @property
    def MIN_START_DATE(self):
        return self._MIN_START_DATE

    @property
    def MAX_END_DATE(self):
        return self._MAX_END_DATE

    @property
    def NW_CORNER_LON_LAT(self):
        return self._NW_CORNER_LON_LAT

    @property
    def HEIGHT(self):
        return self._HEIGHT

    @property
    def WIDTH(self):
        return self._WIDTH

    @property
    def H5_VERSION(self):
        return self._H5_VERSION

    @property
    def MODE(self):
        return self._MODE

    @property
    def CLASSES(self):
        # return self._CLASSES  # TODO: Create a validator for this
        return {'0': (0, 0.1), '0.1': (0.1, 1), '1': (1, 2.5), '2.5': (2.5, np.infty)}

    @property
    def VSC(self):
        return self._VSC

    @property
    def VIDEO_START(self):
        return self._VIDEO_START

    @property
    def VIDEO_END(self):
        return self._VIDEO_END


def check_config_min_max_dates(min_start_date, max_end_date):
    assert min_start_date < max_end_date
    expected_min_start_date = dt.datetime(2005, 6, 1, 0, 50)
    expected_max_start_date = dt.datetime(2020, 12, 31, 23, 50)
    try:
        assert min_start_date >= expected_min_start_date
        assert max_end_date <= expected_max_start_date
    except AssertionError:
        warnings.warn(f"Max and Min dates are outside the expected default "
                      f"range {expected_min_start_date} - {expected_max_start_date}. "
                      f"I got {min_start_date} - {max_end_date} instead.")


class Config:
    already_instantiated = False

    def __init__(self, cfg_content: RadolanConfigFileContent, inside_initialize: bool = False):
        check_date_format(cfg_content)
        self._RANGES_DATE_FORMAT = cfg_content.RANGES_DATE_FORMAT
        self._TIMESTAMP_DATE_FORMAT = cfg_content.TIMESTAMP_DATE_FORMAT

        self._BASE_URL = cfg_content.BASE_URL

        self._RADOLAN_ROOT = os.path.expandvars(os.path.abspath(os.path.expanduser(cfg_content.RADOLAN_ROOT)))

        self._MIN_START_DATE = cfg_content.MIN_START_DATE
        self._MAX_END_DATE = cfg_content.MAX_END_DATE
        check_config_min_max_dates(self._MIN_START_DATE, self._MAX_END_DATE)

        self._DATE_RANGES_FILE_PATH = os.path.join(os.path.expanduser('~/.radolan_config'), 'DATE_RANGES.yml')
        self._VIDEO_RANGES_FILE_PATH = os.path.join(os.path.expanduser('~/.radolan_config'), 'VIDEO_RANGES.yml')
        self._TEST_SET_RANGES_FILE_PATH = os.path.join(os.path.expanduser('~/.radolan_config'), 'TEST_SET_RANGES.yml')
        self._VALIDATION_SET_RANGES_FILE_PATH = os.path.join(os.path.expanduser('~/.radolan_config'),
                                                             'VALIDATION_SET_RANGES.yml')
        self._TRAINING_SET_RANGES_FILE_PATH = os.path.join(os.path.expanduser('~/.radolan_config'),
                                                           'TRAINING_SET_RANGES.yml')

        self._date_ranges = None
        self._files_list = None
        self._video_ranges = None
        self._test_set_ranges = None
        self._validation_set_ranges = None
        self._training_set_ranges = None

        self._ranges_path_dict = {
            'video_ranges':
                {'path': self.VIDEO_RANGES_FILE_PATH,
                 'template_file_name': 'VIDEO_RANGES_TEMPLATE_DONT_MODIFY.yml',
                 'file_name': 'VIDEO_RANGES.yml'},
            'date_ranges':
                {'path': self.DATE_RANGES_FILE_PATH,
                 'template_file_name': 'DATE_RANGES_TEMPLATE_DONT_MODIFY.yml',
                 'file_name': 'DATE_RANGES.yml'},
            'test_set_ranges':
                {'path': self.TEST_SET_RANGES_FILE_PATH,
                 'template_file_name': 'TEST_SET_RANGES_TEMPLATE_DONT_MODIFY.yml',
                 'file_name': 'TEST_SET_RANGES.yml'},
            'valid_set_ranges':
                {'path': self.VALIDATION_SET_RANGES_FILE_PATH,
                 'template_file_name': 'VALIDATION_SET_RANGES_TEMPLATE_DONT_MODIFY.yml',
                 'file_name': 'VALIDATION_SET_RANGES.yml'},
            'train_set_ranges':
                {'path': self.TRAINING_SET_RANGES_FILE_PATH,
                 'template_file_name': 'TRAINING_SET_RANGES_TEMPLATE_DONT_MODIFY.yml',
                 'file_name': 'TRAINING_SET_RANGES.yml'},
        }

        self._NW_CORNER_LON_LAT = np.array(cfg_content.NW_CORNER_LON_LAT)
        self._NW_CORNER_INDICES = coords_finder(*self._NW_CORNER_LON_LAT, distances_output=False)

        self._height = cfg_content.HEIGHT
        self._width = cfg_content.WIDTH

        self._current_h5_version = version.Version('v0.0.8')
        self._h5_version = version.Version(cfg_content.H5_VERSION)

        self._mode = cfg_content.MODE
        self._classes = cfg_content.CLASSES

        self._VSC = cfg_content.VSC

        self._VIDEO_START = cfg_content.VIDEO_START
        self._VIDE_END = cfg_content.VIDEO_END

        radolan_grid_ll = utils.cut_square(
            array=wrl.georef.get_radolan_grid(900, 900, wgs84=True),
            height=self._height,
            width=self._width,
            indices_up_left=self._NW_CORNER_INDICES,
        )

        self._radolan_grid_ll_array = radolan_grid_ll  # shape (900, 900, 2)
        self._coordinates_array = np.moveaxis(radolan_grid_ll, -1, 0)  # shape for concatenation. (2, 900, 900)

        if self._current_h5_version < self._h5_version:
            raise VersionTooLargeError(self._current_h5_version, self._h5_version)
        elif self._current_h5_version > self._h5_version:
            warnings.warn("There could be compatibility issues between current supported version {} and fed version {}"
                          "".format(self._current_h5_version, self._h5_version), FutureWarning)

        if Config.already_instantiated:
            warnings.warn(f'There is already an instance of this class. {type(self)}', UserWarning)
        if not inside_initialize:
            raise UserWarning("Please initialize by calling dwd_dl.cfg.initialize()")
        Config.already_instantiated = True

    @property
    def RANGES_DATE_FORMAT(self):
        return self._RANGES_DATE_FORMAT

    @property
    def TIMESTAMP_DATE_FORMAT(self):
        return self._TIMESTAMP_DATE_FORMAT

    @property
    def RADOLAN_ROOT(self):
        return self._RADOLAN_ROOT

    @property
    def RADOLAN_RAW(self):
        if self.VSC:
            root = os.environ['BINFL']
        else:
            root = self.RADOLAN_ROOT
        return os.path.join(root, 'Raw')

    @property
    def RADOLAN_H5(self):
        if self.VSC:
            root = os.environ['BINFL']
        else:
            root = self.RADOLAN_ROOT
        return os.path.join(root, 'H5')

    @property
    def MIN_START_DATE(self):
        return self._MIN_START_DATE

    @property
    def MAX_END_DATE(self):
        return self._MAX_END_DATE

    @property
    def DATE_RANGES_FILE_PATH(self):
        return self._DATE_RANGES_FILE_PATH

    @property
    def TEST_SET_RANGES_FILE_PATH(self):
        return self._TEST_SET_RANGES_FILE_PATH

    @property
    def VALIDATION_SET_RANGES_FILE_PATH(self):
        return self._VALIDATION_SET_RANGES_FILE_PATH

    @property
    def TRAINING_SET_RANGES_FILE_PATH(self):
        return self._TRAINING_SET_RANGES_FILE_PATH

    @property
    def VIDEO_RANGES_FILE_PATH(self):
        return self._VIDEO_RANGES_FILE_PATH

    @property
    def BASE_URL(self):
        return self._BASE_URL

    @property
    def NW_CORNER_LAT_LON(self):
        return self._NW_CORNER_LON_LAT

    @property
    def NW_CORNER_INDICES(self):
        return self._NW_CORNER_INDICES

    @property
    def HEIGHT(self):
        return self._height

    @property
    def WIDTH(self):
        return self._width

    @property
    def CURRENT_H5_VERSION(self):
        return self._current_h5_version

    @property
    def H5_VERSION(self):
        return self._h5_version

    @property
    def MODE(self):
        return self._mode

    @property
    def CLASSES(self):
        return self._classes

    @property
    def VSC(self):
        return self._VSC

    @property
    def VIDEO_START(self):
        return self._VIDEO_START

    @property
    def VIDEO_END(self):
        return self._VIDE_END

    @property
    def date_ranges(self):
        if self._date_ranges is None:
            self._date_ranges = read_ranges(self.DATE_RANGES_FILE_PATH)
        return self._date_ranges

    @property
    def video_ranges(self):
        if self._video_ranges is None:
            self._video_ranges = read_ranges(self.VIDEO_RANGES_FILE_PATH)
        return self._video_ranges

    @property
    def test_set_ranges(self):
        if self._test_set_ranges is None:
            self._test_set_ranges = read_ranges(self.TEST_SET_RANGES_FILE_PATH)
        return self._test_set_ranges

    @property
    def validation_set_ranges(self):
        if self._validation_set_ranges is None:
            self._validation_set_ranges = read_ranges(self.VALIDATION_SET_RANGES_FILE_PATH)
        return self._validation_set_ranges

    @property
    def training_set_ranges(self):
        if self._training_set_ranges is None:
            self._training_set_ranges = read_ranges(self.TRAINING_SET_RANGES_FILE_PATH)
        return self._training_set_ranges

    @property
    def date_timestamps_list(self):
        return self._timestamps_list(self.date_ranges)

    @property
    def video_timestamps_list(self):
        return self._timestamps_list(self.video_ranges)

    @property
    def test_set_timestamps_list(self):
        return self._timestamps_list(self.test_set_ranges)

    @property
    def validation_set_timestamps_list(self):
        return self._timestamps_list(self.validation_set_ranges)

    @property
    def training_set_timestamps_list(self):
        return self._timestamps_list(self.training_set_ranges)

    @staticmethod
    def _timestamps_list(ranges):
        timestamp_list = []
        for date_range in tqdm(ranges):
            format_cache = date_range.date_format
            date_range.switch_date_format(format_='timestamp_date_format')
            timestamp_list.extend([x for x in date_range.str_date_range()])
            date_range.switch_date_format(format_=format_cache)
        return timestamp_list

    @property
    def files_list(self):
        if self._files_list is None:
            date_ranges_files_list = RadolanFilesList(date_ranges=self.date_ranges)
            video_ranges_files_list = RadolanFilesList(date_ranges=self.video_ranges)
            self._files_list = date_ranges_files_list + video_ranges_files_list
            self._files_list.remove_duplicates()
        return self._files_list

    def create_checkpoint_dir(self):
        checkpoint_dir = os.path.join(self.RADOLAN_ROOT, 'Models')
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    def create_checkpoint_path_with_name(self, experiment_timestamp_str):
        checkpoint_name = self._create_raw_checkpoint_name(experiment_timestamp_str)
        checkpoint_dir = self.create_checkpoint_dir()
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        return checkpoint_path

    @staticmethod
    def _create_raw_checkpoint_name(experiment_timestamp_str):
        """Change prefix if you want to change the checkpoint name.

        """
        prefix = ''
        return prefix + f"{experiment_timestamp_str}.ckpt"

    @property
    def coordinates_array(self):
        return self._coordinates_array

    @property
    def radolan_grid_ll_array(self):
        return self._radolan_grid_ll_array

    def check_and_make_dir_structures(self):
        for dir_ in (self.RADOLAN_ROOT, self.RADOLAN_RAW, self.RADOLAN_H5):
            if not os.path.isdir(dir_):
                os.makedirs(dir_)

    def make_date_ranges(self):
        self._make_single_range(self._ranges_path_dict['date_ranges'])

    def make_video_ranges(self):
        self._make_single_range(self._ranges_path_dict['video_ranges'])

    def make_all_ranges(self):
        for range_ in self._ranges_path_dict:
            self._make_single_range(range_dict=self._ranges_path_dict[range_])

    @staticmethod
    def _make_single_range(range_dict):
        if not os.path.isfile(range_dict['path']):
            template_file_path = path_to_resources_folder(range_dict['template_file_name'])
            shutil.copy2(template_file_path, range_dict['path'], follow_symlinks=False)
            print(f"Created {range_dict['file_name']} in {range_dict['path']}.")
        else:
            print(f"{range_dict['path']} already exists. Just edit it!")

    def check_downloaded_files(self):
        # compare with existing
        missing_files = RadolanFilesList(files_list=[file for file in self.files_list if not file.exists()])
        if not missing_files:
            print("No missing files!")
        else:
            print(f"{len(missing_files)} missing files!")

        return missing_files

    def download_missing_files(self):
        missing_files = self.check_downloaded_files()

        if not missing_files:
            input_message = "Nothing to download. "
        else:
            total_size = missing_files.total_download_size
            input_message = f'Total size is {total_size} bytes. '

        while True:
            try:
                x = input(input_message + 'Proceed? y/[n] ')
            except EOFError:
                x = 'y'
            if x in ('y', 'Y'):
                break
            sys.exit()

        if missing_files:
            with tempfile.TemporaryDirectory() as td:
                print('Creating Temporary Directory')
                if os.path.isdir(os.path.abspath(td)):
                    print(f'Temporary Directory created: {os.path.abspath(td)}')
                else:
                    raise OSError("The temporary directory was not created.")

                download_files_to_directory(
                    os.path.abspath(td),
                    missing_files.download_list,
                )

                listdir = os.listdir(td)
                for file in listdir:
                    if file.endswith('.tar.gz'):
                        with tarfile.open(os.path.join(td, file), 'r:gz') as tf:
                            print(f'Extracting all in {file}.')
                            tf.extractall(self.RADOLAN_RAW)
                            print('All extracted.')
                    elif file.endswith('.gz'):
                        with gzip.open(
                                os.path.join(td, file), 'rb'
                        ) as f_in, open(
                            os.path.join(self.RADOLAN_RAW, file.replace('.gz', '')), 'wb'
                        ) as f_out:
                            shutil.copyfileobj(f_in, f_out)

                print("Extracting compressed day files.")
                for file in tqdm(os.listdir(self.RADOLAN_RAW)):
                    if file.endswith('.gz'):
                        with gzip.open(
                            os.path.join(self.RADOLAN_RAW, file), 'rb'
                        ) as f_in, open(
                            os.path.join(self.RADOLAN_RAW, file.replace('.gz', '')), 'wb'
                        ) as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        os.remove(os.path.join(self.RADOLAN_RAW, file))
                print("Done.")

    def validate_all_ranges(self):
        for range_list_1, range_list_2 in itertools.combinations(
                (self.training_set_ranges, self.validation_set_ranges, self.test_set_ranges), 2
        ):
            for date_range_1 in range_list_1:
                for date_range_2 in range_list_2:
                    assert date_range_1.end < date_range_2.start or date_range_1.start > date_range_2.end

        for range_list in (self.training_set_ranges, self.validation_set_ranges, self.test_set_ranges):
            for date_range in range_list:
                found = False
                for date_range_all in self.date_ranges:
                    if date_range.start >= date_range_all.start and date_range.end <= date_range_all.end:
                        found = True
                if not found:
                    raise ValueError(f"{date_range} not found in {date_range_all}")

    def get_timestamps_hash(self):
        raise DeprecationWarning

    def check_h5_file(self):
        raise NotImplementedError

    def add_missing_to_h5(self):
        raise NotImplementedError


class RadolanFilesList:
    def __init__(self, date_ranges=None, files_list=None):
        self.files_list = []
        if date_ranges and files_list:
            raise ValueError(f"Got both ranges_list = {date_ranges} and file_list = {files_list}")
        elif date_ranges:
            year_month_tuples = utils.ym_tuples(date_ranges)
            for ym in year_month_tuples:
                self.files_list += [RadolanFile(date) for date in MonthDateRange(*ym).date_range()]
        elif files_list:
            self._valid_inputs_for_files_list_add(files_list)
            self.__iadd__(files_list)

        # total download size initialization
        self._total_size = 0

        self._download_list = None

    @staticmethod
    def _valid_inputs_for_files_list_add(other):
        if not (isinstance(other, RadolanFilesList) or isinstance(other, list)):
            raise TypeError(f"Expected either list of RadolanFile or RadolanFilesList but got {type(other)} "
                            f"instead")
        for file in other:
            if not isinstance(file, RadolanFile):
                raise TypeError(f"Expected RadolanFile in {other} but got {type(file)}")

    def __add__(self, other):
        self._valid_inputs_for_files_list_add(other)
        new_list = self.files_list + [element for element in other]
        return RadolanFilesList(files_list=new_list)

    def __iadd__(self, other):
        self._valid_inputs_for_files_list_add(other)
        self.files_list += [element for element in other]

        # resetting total size
        self._total_size = 0

    def __contains__(self, item):
        return item in self.files_list

    def __iter__(self):
        return iter(self.files_list)

    def __len__(self):
        return len(self.files_list)

    def remove_duplicates(self):
        self.files_list = list(dict.fromkeys(self.files_list))

    @property
    def download_list(self):
        if not self._download_list:
            download_list = []
            print("Computing download list.")
            for file in self.files_list:
                if file.date not in download_list:
                    download_file = DownloadFile(file.year, file.month, file.date, CFG.BASE_URL)
                    download_list.append(download_file)
                    print(f"Added {download_file} to the download list.")
            self._download_list = download_list
            print("Done")
        else:
            download_list = self._download_list
        return download_list

    @property
    def total_download_size(self):

        if not self._total_size == 0:
            return self._total_size

        total_size = 0

        print("Computing download size.")
        for file in self.download_list:
            total_size += file.size

        self._total_size = total_size

        return self._total_size


class RadolanFile:
    def __init__(self, date):
        self.date = date

    @property
    def file_name(self):
        return binary_file_name(self.date)

    @property
    def year(self):
        return self.date.year

    @property
    def month(self):
        return self.date.month

    def get_relevant_file_to_download(self):
        print(f"Getting relevant file to download for {self.year} - {self.month} - {self.date}")
        return get_download_url(self.year, self.month, self.date, base_url=CFG.BASE_URL)

    def get_file_name(self):
        return get_download_file_name(self.year, self.month, self.date)

    def __eq__(self, other):
        if isinstance(other, RadolanFile):
            return other.file_name == self.file_name
        elif isinstance(other, str):
            return other == self.file_name

    def __str__(self):
        return self.file_name

    def __hash__(self):
        return hash(self.file_name)

    def exists(self):
        if not os.path.isfile(os.path.join(CFG.RADOLAN_RAW, self.file_name)):
            try:
                for file_name in binary_file_name_approx_generator(self.date):
                    file_path = os.path.join(CFG.RADOLAN_RAW, file_name)
                    if os.path.isfile(file_path):
                        return True
            except OverflowError:
                print(f"File {self.file_name} not found even with approximation loop. Sorry.")
                return False
        return True


def get_download_size(url):
    r = requests.head(url)
    print(f"Getting download size for: {url}")
    return int(r.headers['Content-Length'])


def initialize(inside_initialize=True, skip_download=False):
    global CFG
    if CFG is not None and not isinstance(CFG, Config):  # The condition after and is redundant. For readability.
        raise TypeError(
            "Expected type {} but got {} of type {}. CFG was tampered with.".format(type(Config), CFG, type(CFG))
        )

    cfg_content = read_or_make_config_file()
    radolan_configurator = Config(cfg_content, inside_initialize=inside_initialize)
    CFG = radolan_configurator
    CFG.check_and_make_dir_structures()
    CFG.make_all_ranges()
    check_ranges_overlap(CFG.date_ranges)
    check_ranges_overlap(CFG.training_set_ranges)
    check_ranges_overlap(CFG.validation_set_ranges)
    check_ranges_overlap(CFG.test_set_ranges)
    check_ranges_overlap(CFG.video_ranges)
    CFG.validate_all_ranges()
    if not skip_download:
        if (ds.check_datasets_missing_or_corrupt(CFG.date_ranges, classes=CFG.CLASSES) or
                ds.check_datasets_missing_or_corrupt(CFG.video_ranges, classes=CFG.CLASSES)):
            CFG.download_missing_files()
    os.environ['WRADLIB_DATA'] = CFG.RADOLAN_RAW
    return CFG


def read_or_make_config_file():
    try:
        cfg_content = read_radolan_config_file()
    except FileNotFoundError:
        cfg_content = make_radolan_config_file()
    return cfg_content


def read_radolan_config_file() -> RadolanConfigFileContent:
    data = yu.load_config(os.path.join(os.path.expanduser('~/.radolan_config'), 'RADOLAN_CFG.yml'))
    yu.validate_config(data)
    radolan_config_file_content = RadolanConfigFileContent(**data[0][0])
    return radolan_config_file_content


def make_radolan_config_file():
    """Makes a radolan cfg file froma a template in the resources folder.

    The radolan configuration file is saved in the ~/.radolan_config/ folder.
    """
    radolan_cfg_dir = os.path.join(os.path.expanduser('~'), '.radolan_config')
    if not os.path.isdir(radolan_cfg_dir):
        os.makedirs(radolan_cfg_dir)
    radolan_config_file_name = 'RADOLAN_CFG.yml'
    template_file_name = 'RADOLAN_CFG_TEMPLATE_DONT_MODIFY.yml'

    template_file_path = path_to_resources_folder(template_file_name)

    shutil.copy2(template_file_path, os.path.join(radolan_cfg_dir, radolan_config_file_name))

    return read_radolan_config_file()


def path_to_resources_folder(filename=None):
    installation_path = os.path.abspath(os.path.dirname(inspect.getfile(dl)))
    resources_path = os.path.join(installation_path, 'resources')
    if filename:
        return os.path.join(resources_path, filename)
    else:
        return resources_path


def check_date_format(cfg_content: RadolanConfigFileContent):
    stamps_ranges = ('%Y', '%m', '%d', '%H', '%M', '%S')
    assert all(stamp in cfg_content.RANGES_DATE_FORMAT for stamp in stamps_ranges)

    stamps_timestamps = ('%y', '%m', '%d', '%H', '%M')
    assert all(stamp in cfg_content.TIMESTAMP_DATE_FORMAT for stamp in stamps_timestamps)


def check_connection(url: str) -> bool:
    """A function to check if a website is up.

    Parameters
    ----------
    url : str
        URL to which the connection should be checked.

    Returns
    -------
    bool
        True if the site is up. Else False.

    """
    site_is_up_code = 200
    return urllib.request.urlopen(url).getcode() == site_is_up_code


def check_date_ranges():
    raise NotImplementedError


class DownloadFile:
    def __init__(self, year, month, date, base_url):
        assert date.year == year
        assert date.month == month
        file_name = get_monthly_file_name(year, month, date, with_name_discrepancy=True)
        url = base_url + f'{year}/' + file_name
        size = file_is_available(url)
        is_monthly = True
        if not size:
            file_name = binary_file_name(date, extension='.gz')
            url = base_url.replace('historical', 'recent') + file_name
            size = file_is_available(url)
            is_monthly = False

        self._url = url
        self._file_name = file_name
        self._size = size
        self._is_monthly = is_monthly
        self._year = year
        self._month = month
        if self._is_monthly:
            self._date = dt.datetime(year, month, 1, 0, 50)
        else:
            self._date = date

    @property
    def url(self):
        return self._url

    @property
    def file_name(self):
        return self._file_name

    @property
    def size(self):
        return self._size

    @property
    def date(self):
        return self._date

    @property
    def year(self):
        return self._year

    @property
    def month(self):
        return self._month

    def is_monthly(self):
        return self._is_monthly

    def is_hourly(self):
        return not self._is_monthly

    def contains_date(self, date: dt.datetime):
        assert isinstance(date, dt.datetime)
        if self.is_hourly():
            return self.date == date
        else:
            return self.year == date.year and self.month == date.month

    def __eq__(self, other):
        if not (isinstance(other, DownloadFile) or isinstance(other, dt.datetime)):
            raise TypeError(f"Expected {type(DownloadFile)} or {type(dt.datetime)} but got {type(other)} instead.")
        if isinstance(other, DownloadFile):
            return self.file_name == other.file_name
        else:
            return self.contains_date(other)

    def __str__(self):
        return f"<DownloadFile named: {self.file_name} of size: {self.size}.>"


def get_monthly_file_name(year, month, date, with_name_discrepancy=False):
    if date.year >= 2020:
        warnings.warn("The Month File is probably not available for year {} month {}".format(year, month))
    if year == 2005 or not with_name_discrepancy:
        return 'RW-{}{:02d}.tar.gz'.format(year, month)
    else:
        return 'RW{}{:02d}.tar.gz'.format(year, month)


def get_download_url(year, month, date: dt.datetime, base_url):  # TODO: refactor this
    url = base_url + f'{year}/' + get_monthly_file_name(year, month, date, with_name_discrepancy=True)
    if not file_is_available(url=url):
        url = base_url.replace('historical', 'recent') + binary_file_name(date, extension='.gz')
    return url


def get_download_file_name(year, month, date: dt.datetime):
    if 'historical' in get_download_url(year, month, date, base_url=CFG.BASE_URL):
        return get_monthly_file_name(year, month, date)
    else:
        return binary_file_name(date, extension='.gz')


def read_ranges(ranges_path):
    print('Reading ranges.')

    date_ranges_data = yu.load_ranges(ranges_path)
    yu.validate_ranges(date_ranges_data)
    date_ranges = [DateRange(start_date, end_date) for start_date, end_date in date_ranges_data[0][0]]
    print('Finished reading.')

    return date_ranges


def check_ranges_overlap(ranges_list):
    for first_date_range, second_date_range in itertools.combinations(ranges_list, 2):
        if not first_date_range.end < second_date_range.start:  # all other conditions follow since start < end always.
            raise ValueError("There is an overlap or wrong order in date ranges. {} {} and {} {}.".format(
                first_date_range.start, first_date_range.end, second_date_range.start, second_date_range.end
            ))


class DateRange:
    def __init__(self, start_date, end_date, date_format=None):
        if not date_format:
            date_format = CFG.RANGES_DATE_FORMAT
        self._date_format = date_format
        start = start_date
        end = end_date
        if start > end:
            raise ValueError('Start date is bigger than end date.')
        self.start = start
        self.end = end

    def __iter__(self):
        return iter([self.start, self.end])

    def __str__(self):
        return "{}_{}".format(self.start.strftime(self._date_format), self.end.strftime(self._date_format))

    def __contains__(self, item):
        if isinstance(item, dt.datetime):
            return self.start <= item <= self.end
        return False

    def switch_date_format(self, format_=None):
        assert format_ in ('ranges_date_format', 'timestamp_date_format', None,
                           CFG.TIMESTAMP_DATE_FORMAT, CFG.RANGES_DATE_FORMAT,)

        if (format_ == 'ranges_date_format' or (format_ is None and self._date_format == CFG.TIMESTAMP_DATE_FORMAT)
                or format_ == CFG.RANGES_DATE_FORMAT):
            self._date_format = CFG.RANGES_DATE_FORMAT
        elif (format_ == 'timestamp_date_format' or (format_ is None and self._date_format == CFG.RANGES_DATE_FORMAT)
                or format_ == CFG.TIMESTAMP_DATE_FORMAT):
            self._date_format = CFG.TIMESTAMP_DATE_FORMAT
        return self

    @property
    def date_format(self):
        return self._date_format

    def date_range(self, include_end=True):
        return list(daterange(self.start, self.end, include_end=include_end))

    def str_date_range(self, include_end=True):
        assert self._date_format is not None
        return [x.strftime(self.date_format) for x in self.date_range(include_end=include_end)]


class MonthDateRange(DateRange):
    def __init__(self, year: int, month: int):
        start_date = dt.datetime(year=year, month=month, day=1, hour=0, minute=50)
        year, month = utils.next_year_month(year, month)
        end_date = dt.datetime(year=year, month=month, day=1, hour=0, minute=50) - dt.timedelta(hours=1)
        super().__init__(start_date=start_date, end_date=end_date)


def file_is_available(url):
    """Checks if a DWD file is available by looking at the HTTP header.

    Only the HTTP header is requested. The file is not downloaded. Returns the number of bytes as an integer.

    Parameters
    ----------
    url : str
        Valid url as a string.

    Returns
    -------
    int
        Number of bytes of the requested files. Returns 0 if the file was not reachable.

    """
    r = requests.head(url)
    if not r.headers['Content-Type'] == 'application/octet-stream':
        return 0
    else:
        return int(r.headers['Content-Length'])


def download_files_to_directory(dirpath, downloadable_list: List[DownloadFile]):
    """Downloads files to directory.

    Parameters
    ----------
    dirpath : str
        Valid path to a directory.
    downloadable_list : str or list of str
        Valid urls pointing to downloadable files.

    Raises
    ------
    OSError
        Error is raised if the given dirpath is invalid according to os.path.isdir().

    """
    if not os.path.isdir(dirpath):
        raise OSError('Invalid Path.')

    for file in downloadable_list:
        print(f'Downloading file {file.file_name}')
        r = requests.get(file.url)
        with open(os.path.join(os.path.abspath(dirpath), file.file_name), 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)


def interval_is_valid(start_datetime, end_datetime):
    """Checks if a given time intervall is valid for DWD operations.

    The permitted intervall is `2005-06-01 00:50` to `2019-12-31 23:50`.

    Parameters
    ----------
    start_datetime, end_datetime : datetime.datetime
        Start and end timestamps.

    Returns
    -------
    bool
        Tells whether the input is valid or not.

    """
    if start_datetime < dt.datetime(2005, 6, 1, 0, 50) or end_datetime > dt.datetime(2019, 12, 31, 23, 50):
        return False
    else:
        return True


def clean_unused():
    """Cleans unused files from RADOLAN_PATH

    """

    assert config_was_run()

    listdir = os.listdir(os.path.abspath(CFG.RADOLAN_ROOT))
    date_ranges = read_ranges(CFG.DATE_RANGES_FILE_PATH)

    for file in tqdm(listdir):
        if 'dwd---bin' not in file:
            continue
        file_used = False
        for date_range in date_ranges:
            if file in used_files(date_range.start, date_range.end):
                file_used = True
        if not file_used:
            os.remove(os.path.join(os.path.abspath(CFG.RADOLAN_ROOT), file))


def daterange(start_date, end_date, include_end=False):
    """A useful date range generator.

    Parameters
    ----------
    start_date : datetime.datetime
        A start date.
    end_date : datetime.datetime
        An end date.
    include_end : bool
        Tells if the generator should include the end_date.

    Yields
    ------
    A sequence of datetime.datetime objects equally spaced in hours. (Actually not, but for the time being this
    description is sufficient).

    """
    if include_end:
        end = 1
    else:
        end = 0
    for n in range(int((end_date - start_date).total_seconds() / 3600) + end):
        yield start_date + dt.timedelta(hours=n)


def used_files(start_date, end_date):
    """Yields the name of the used files

    Parameters
    ----------
    start_date : datetime.datetime
        A valid start date.
    end_date : datetime.datetime
        A valid end date.

    Yields
    ------
    str
        File names of the DWD binaries actually used.

    """
    for date in daterange(start_date, end_date, include_end=True):
        yield binary_file_name(date)


def check_date_ranges_path(fpath, fname=None):
    if not os.path.isfile(os.path.abspath(fpath)) and not os.path.isfile(os.path.join(os.path.abspath(fpath), fname)):
        raise OSError('Invalid location for {}'.format(fname))


def config_was_run():
    """Checks if dwd_dl.config.config_initializer() was run by looking at global variables.

    Returns
    -------
    bool
        Straightforward output.

    """
    global CONFIG_WAS_RUN
    return CONFIG_WAS_RUN


def binary_file_name(time_stamp, extension=None):
    """Returns the file name of a DWD binary given a datetime.datetime timestamp.

    Parameters
    ----------
    extension : str
        A string that is appended to the filename.
    time_stamp : datetime.datetime
        A valid timestamp.

    Returns
    -------
    str
        The name of the binary corresponding to the given timestamp. `raa01-rw_10000-yymmDDHHMM-dwd---bin`

    """
    file_name = 'raa01-rw_10000-{}-dwd---bin'.format(time_stamp.strftime(CFG.TIMESTAMP_DATE_FORMAT))
    if extension:
        return file_name + extension
    return file_name


def binary_file_name_approx_generator(time_stamp, extension=None):
    """

    Parameters
    ----------
    time_stamp
    extension

    Yields
    ------

    """
    for i in range(60):
        delta = dt.timedelta(minutes=((-1)**i*((i+1)//2)))  # gives: 0 -1 1 -2 2 and so on
        yield binary_file_name(time_stamp + delta, extension=extension)
    raise OverflowError


def distance(a_tup, b_tup):
    """Computes the distance between numpy.ndarray .

    Parameters
    ----------
    a_tup : numpy.ndarray
        Preferably a simple (y,x) coordinate as numpy.ndarray.
    b_tup : numpy.ndarray
        A 3D numpy.ndarray with a dimension being 2. Broadcasting rules apply.

    Returns
    -------
    numpy.ndarray
        A 2D array containing the distances of each and every point in b_tup from a_tup.

    """
    return np.sqrt((np.subtract(a_tup, b_tup) ** 2).sum(axis=2))


def coords_finder(lon, lat, distances_output=False, verbose=False):
    """finds x, y coordinates given lon lat for the 900x900 RADOLAN grid.

    Parameters
    ----------
    lat : float
        A valid latitude.
    lon : float
        A valid longitude.
    distances_output : bool
        Whether the function shall return the dwd_dl.config.distance() output as well.

    Returns
    -------
    tuple
        A tuple containing the indeces of the DWD Radolan Grid point nearest to the supplied coordinates.
    numpy.ndarray, optional
        The output of dwd_dl.config.distances()

    See Also
    --------
    dwd_dl.config.distances()

    """
    proj_stereo = wrl.georef.create_osr("dwd-radolan")
    if verbose:
        print(proj_stereo)
    proj_wgs = osr.SpatialReference()
    proj_wgs.ImportFromEPSG(4326)
    if verbose:
        print(proj_wgs)
    coords_ll = np.array([lon, lat])
    radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
    coords_xy = wrl.georef.reproject(coords_ll, projection_source=proj_wgs, projection_target=proj_stereo)
    # TODO: refactor line above using wrl.georef.get_radolan_coords

    distances = distance(coords_xy, radolan_grid_xy)

    if distances_output:
        return np.unravel_index(distances.argmin(), distances.shape), distances
    else:
        return np.unravel_index(distances.argmin(), distances.shape)


class VersionError(Exception):
    """To be raised when there is a version mismatch. """

    def __init__(self, v1: version.Version, v2: version.Version, message="Version {} is incompatible with version {}"):
        self.message = message.format(v1, v2)
        super().__init__(self.message)


class VersionTooLargeError(VersionError):
    """To be raised when the expected version is too large."""

    def __init__(self, v_expected: version.Version, v_given: version.Version):
        self.v_expected = v_expected
        self.v_given = v_given
        super().__init__(v1=v_expected, v2=v_given)

    def __str__(self):
        return f"{self.message}. Given: {self.v_given} is greater than expected: {self.v_expected}."


class VersionTooSmallError(VersionError):
    """To be raised when the expected version is too small."""

    def __init__(self, v_expected: version.Version, v_given: version.Version):
        self.v_expected = v_expected
        self.v_given = v_given
        super().__init__(v1=v_expected, v2=v_given)

    def __str__(self):
        return f"{self.message}. Given: {self.v_given} is less than expected: {self.v_expected}."
