"""Module for checking requirements and downloading the radar data.

"""

import os
import requests
import datetime
import tempfile
import sys
import tarfile
import configparser
import numpy as np
import wradlib as wrl
from osgeo import osr

DOWNLOAD_URL_LIST = []
DOWNLOAD_FILE_NAMES_LIST = []
SIZES_LIST = []
BASE_URL = 'https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/historical/bin/'
DEFAULT_RADOLAN_PATH = os.path.abspath('./Radolan/')
DEFAULT_START_DATE = datetime.datetime(2005, 6, 1, 0, 50)
DEFAULT_END_DATE = datetime.datetime(2019, 12, 31, 23, 50)

# TODO: Add download manager for current year


def radar_download_list_generator(start_year=DEFAULT_START_DATE.year, start_month=DEFAULT_START_DATE.month,
                                  end_year=DEFAULT_END_DATE.year, end_month=DEFAULT_END_DATE.month):
    """Download lists generator. Outputs urls, names and sizes.

    Parameters
    ----------
    start_year, start_month, end_year, end_month : int
        The start and end year and month of the period to be downloaded. (The default is derived from the module
        variables `DEFAULT_START_DATE` and `DEFAULT_END_DATE` which are 2005-06-01 00:50 and 2019-12-31 23:50
        respectively.)

    Returns
    -------
    download_list : list of str
        The list of the urls pointing to the `.tar.gz` files to be downloaded.
    file_names_list : list of str
        A list containing the names of the files to be downloaded.
    sizes_list : list of int
        A list of the sizes of the files to be donwloaded. Element is 0 if the file wasn't available.
    """
    download_list = []
    file_names_list = []
    sizes_list = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if not (year == start_year and month < start_month) and not (year == end_year and month > end_month):
                if year == 2005:
                    url = BASE_URL + f'{year}/' + 'RW-{}{:02d}.tar.gz'.format(year, month)
                else:
                    url = BASE_URL + f'{year}/' + 'RW{}{:02d}.tar.gz'.format(year, month)

                if file_is_available(url):
                    download_list.append(url)
                    file_names_list.append('RW-{}{:02d}.tar.gz'.format(year, month)) # TODO: Review. Possible confilcts.
                    sizes_list.append(file_is_available(url))

    return download_list, file_names_list, sizes_list


def cfg_list(path):
    """Simple file reader.

    Parameters
    ----------
    path : str
        A valid path pointing to the file to be read.

    Returns
    -------
    radar_files : list of str
        A list of the lines read from the file.

    Notes
    -----
    Probably useless.

    """
    with open(path, 'r') as f:
        radar_files = f.read().splitlines()
    return radar_files


def make_cfg(path_list, path_names, path_sizes, *args):
    """Creates files containing the output of dwd_dl.config.radar_download_list_generator().

    Parameters
    ----------
    path_list, path_names, path_sizes : str
        Valid paths to save dwd_dl.config.radar_download_list_generator() outputs.
    args
        Inputs for dwd_dl.config.radar_download_list_generator().

    Returns
    -------
    bool
        True if operation was successful.

    See Also
    --------
    dwd_dl.config.radar_download_list_generator()
    """
    with open(path_list, 'w+') as f1, open(path_names, 'w+') as f2, open(path_sizes, 'w+') as f3:
        download_list, file_names_list, sizes_list = radar_download_list_generator(*args)
        f1.writelines(download_list)
        f2.writelines(file_names_list)
        f3.writelines(sizes_list)
    return True


def cfg_handler(*args, path=os.path.abspath('.')):
    """A (not so) intuitive configuration handler. Even deprecated. Too bad.

    The handler saves three different `.CFG` files in the supplied path containing the output of
    dwd_dl.config.radar_download_list_generator() .

    Parameters
    ----------
    args
        Inputs to dwd_dl.config.make_cfg().
    path : str
        Valid path for saving the CFG files. (Defaults to pwd.)

    Returns
    -------
    None

    Notes
    -----
    Deprecated. Will be removed.
    """
    if ((not os.path.isfile(os.path.join(path, 'RADOLAN_DOWNLOADS.cfg')))
            and (not os.path.isfile(os.path.join(path, 'RADOLAN_NAMES.cfg')))
            and (not os.path.isfile(os.path.join(path, 'RADOLAN_SIZES.cfg')))):
        make_cfg(os.path.join(path, 'RADOLAN_DOWNLOADS.cfg'),
                 os.path.join(path, 'RADOLAN_NAMES.cfg'),
                 os.path.join(path, 'RADOLAN_SIZES.cfg'),
                 *args)

    # TODO: check if file is present, then output what's inside. Give choice to override.
    # TODO: revise global variables and cfg handling.
    # TODO: different cfg handling. Parameters in file.

    global DOWNLOAD_URL_LIST, DOWNLOAD_FILE_NAMES_LIST, SIZES_LIST
    DOWNLOAD_URL_LIST = cfg_list(os.path.join(path, 'RADOLAN_DOWNLOADS.cfg'))
    DOWNLOAD_FILE_NAMES_LIST = cfg_list(os.path.join(path, 'RADOLAN_NAMES.cfg'))
    SIZES_LIST = cfg_list(os.path.join(path, 'RADOLAN_SIZES.cfg'))


def file_is_available(url):
    r = requests.head(url)
    if not r.headers['Content-Type'] == 'application/octet-stream':
        return 0
    else:
        return int(r.headers['Content-Length'])


def download_files_to_directory(dirpath, downloadable_list, names_list):
    if not os.path.isdir(dirpath):
        raise OSError('Invalid Path.')
    if type(downloadable_list) is not list:
        downloadable_list = [downloadable_list]

    for url, name in zip(downloadable_list, names_list):
        print(f'Downloading file {name}')
        r = requests.get(url)
        with open(os.path.join(os.path.abspath(dirpath), name), 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)


def interval_is_valid(start_datetime, end_datetime):
    if start_datetime < datetime.datetime(2005, 6, 1, 0, 50) or end_datetime > datetime.datetime(2019, 12, 31, 23, 50):
        return False
    else:
        return True


def download_and_extract():
    if not config_was_run():
        config_initializer()

    download_list, file_names_list, sizes_list = radar_download_list_generator(
        start_year=START_DATE.year,
        start_month=START_DATE.month,
        end_year=END_DATE.year,
        end_month=END_DATE.month
    )

    total_size = 0

    for element in sizes_list:
        total_size += element


    while True:
        x = input(f'Total size is {total_size} bytes. Proceed? y/[n] ')
        if x in ('y', 'Y'):
            break
        sys.exit()

    # TODO: Better comparison between dates. Remove not needed files.

    with tempfile.TemporaryDirectory() as td:
        print('Creating Temporary Directory')
        print(f'Temporary Directory created: {os.path.isdir(os.path.abspath(td))}')
        download_files_to_directory(os.path.abspath(td), download_list, file_names_list)
        listdir = os.listdir(td)
        for file in listdir:
            if not file.endswith('.tar.gz'):
                continue
            with tarfile.open(os.path.join(td, file), 'r:gz') as tf:
                print(f'Extracting all in {file}.')
                tf.extractall(os.path.abspath(RADOLAN_PATH))
                print('All extracted.')


def clean_unused():
    """Cleans unused files from RADOLAN_PATH"""

    listdir = os.listdir(os.path.abspath(RADOLAN_PATH))

    for file in listdir:
        if not file in used_files(START_DATE, END_DATE):
            os.remove(os.path.join(os.path.abspath(RADOLAN_PATH), file))


def daterange(start_date, end_date, include_end=False):
    if include_end:
        end = 1
    else:
        end = 0
    for n in range(int((end_date - start_date).seconds/3600) + end):
        yield start_date + datetime.timedelta(hours=n)


def used_files(start_date, end_date):
    """Yields the name of the used files"""
    for date in daterange(start_date, end_date, include_end=True):
        yield binary_file_name(date)


def config_initializer(config_fpath='.'):
    """Cheks if global variables are set and are viable. Otherwise it checks for a .cfg file. It creates one if it
    doesn't exist.

    """

    globals_to_check = ['RADOLAN_PATH', 'START_DATE', 'END_DATE']
    config_fname = 'RADOLAN.cfg'

    config = configparser.ConfigParser()
    try:
        config.read(os.path.join(os.path.abspath(config_fpath), config_fname))
    except TypeError:
        for var in globals_to_check:
            config['DEFAULT'][var] = str(globals()['DEFAULT_' + var])
            globals()[var] = globals()['DEFAULT_' + var]

    for var in globals_to_check:
        if var not in config['DEFAULT']:
            config['DEFAULT'][var] = str(globals()['DEFAULT_' + var])
            globals()[var] = globals()['DEFAULT_' + var]

    if not os.path.isdir(os.path.abspath(config['DEFAULT'][globals_to_check[0]])):
        error_message = 'Invalid RADOLAN_PATH. Resorting to default path: ' + DEFAULT_RADOLAN_PATH
        print(error_message)
        config['DEFAULT'][globals_to_check[0]] = str(globals()['DEFAULT_' + globals_to_check[0]])
        globals()[globals_to_check[0]] = globals()['DEFAULT_' + globals_to_check[0]]
    else:
        globals()['RADOLAN_PATH'] = config['DEFAULT'][globals_to_check[0]]

    for date in globals_to_check[1:]:
        try:
            globals()[date] = datetime.datetime.strptime(config['DEFAULT'][date], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            error_message = 'Wrong date format for {}. Expected format is: %Y-%m-%d %H:%M:%S. ' \
                            'Resorting to default value for {}: {}'.format(date, date, str(globals()['DEFAULT_' + date]))
            print(error_message)
            config['DEFAULT'][date] = str(globals()['DEFAULT_' + date])
            globals()[date] = globals()['DEFAULT_' + date]

        if not (globals()['DEFAULT_START_DATE'] <= globals()[date] <= globals()['DEFAULT_END_DATE']):
            error_message = '{} {} should be between {} and {}. Resorting to default value: {}.'.format(
                date,
                str(globals()[date]),
                str(globals()['DEFAULT_START_DATE']),
                str(globals()['DEFAULT_END_DATE']),
                str(globals()['DEFAULT_' + date])
            )
            print(error_message)
            config['DEFAULT'][date] = str(globals()['DEFAULT_' + date])
            globals()[date] = globals()['DEFAULT_' + date]

    if globals()['START_DATE'] > globals()['END_DATE']:
        error_message = 'START_DATE {} is greater than END_DATE {}. Swapping values.'.format(
            str(globals()['START_DATE']),
            str(globals()['END_DATE'])
        )
        globals()['START_DATE'], globals()['END_DATE'] = globals()['END_DATE'], globals()['START_DATE']  # Swapping

    for var in globals_to_check:
        config['DEFAULT'][var] = str(globals()[var])

    with open(os.path.join(os.path.abspath(config_fpath), config_fname), 'w') as configfile:
        config.write(configfile)


def config_was_run():
    try:
        assert START_DATE
        assert END_DATE
        assert RADOLAN_PATH
    except AttributeError:
        return False
    else:
        return True


def binary_file_name(time_stamp):
    return 'raa01-rw_10000-{}-dwd---bin'.format(time_stamp.strftime('%y%m%d%H%M'))


def distance(a_tup, b_tup):
    return np.sqrt((np.subtract(a_tup, b_tup)**2).sum(axis=2))


def coords_finder(lat, lon, distances_output=False):
    """finds x, y coordinates given lon lat for the 900x900 RADOLAN grid

    """
    proj_stereo = wrl.georef.create_osr("dwd-radolan")
    print(proj_stereo)
    proj_wgs = osr.SpatialReference()
    proj_wgs.ImportFromEPSG(4326)
    print(proj_wgs)
    coords_ll = np.array([lat, lon])
    radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
    coords_xy = wrl.georef.reproject(coords_ll, projection_source=proj_wgs, projection_target=proj_stereo)
    # TODO: refactor line above using wrl.georef.get_radolan_coords

    distances = distance(coords_xy, radolan_grid_xy)

    if distances_output:
        return np.unravel_index(distances.argmin(), distances.shape), distances
    else:
        return np.unravel_index(distances.argmin(), distances.shape)
