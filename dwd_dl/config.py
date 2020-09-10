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
from tqdm import tqdm

DOWNLOAD_URL_LIST = []
DOWNLOAD_FILE_NAMES_LIST = []
SIZES_LIST = []
BASE_URL = 'https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/historical/bin/'
DEFAULT_RADOLAN_PATH = os.path.abspath('./Radolan/')
DATE_RANGES_PATH = None
RADOLAN_PATH = None
DEFAULT_START_DATE = datetime.datetime(2005, 6, 1, 0, 50)
DEFAULT_END_DATE = datetime.datetime(2019, 12, 31, 23, 50)
CONFIG_WAS_RUN = False
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# TODO: Add download manager for current year


def radar_download_list_generator(date_ranges_path):
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
    date_ranges = read_ranges(date_ranges_path)
    for date_range in tqdm(date_ranges):
        for year in range(date_range.start.year, date_range.end.year + 1):
            for month in range(1, 13):
                if not (year == date_range.start.year and month < date_range.start.month) \
                        and not (year == date_range.end.year and month > date_range.end.month):
                    if year == 2005:
                        url = BASE_URL + f'{year}/' + 'RW-{}{:02d}.tar.gz'.format(year, month)
                    else:
                        url = BASE_URL + f'{year}/' + 'RW{}{:02d}.tar.gz'.format(year, month)

                    if file_is_available(url) and get_file_name(year, month) not in file_names_list:
                        download_list.append(url)
                        file_names_list.append(get_file_name(year, month))  # TODO: Review. Possible confilcts.
                        sizes_list.append(file_is_available(url))

    return download_list, file_names_list, sizes_list


def get_file_name(year, month):
    return 'RW-{}{:02d}.tar.gz'.format(year, month)


def read_ranges(date_ranges_path):

    print('Reading date ranges.')

    print('Opening file containing date ranges.')
    with open(date_ranges_path, 'r') as f:
        lines = f.read().splitlines()

    date_ranges = []
    for line in lines:
        date_ranges.append(DateRange(*line.split(' _ ')))

    print('Finished reading.')

    return date_ranges


class DateRange:
    def __init__(self, start_date, end_date, date_format=DATE_FORMAT):
        start = datetime.datetime.strptime(start_date, date_format)
        end = datetime.datetime.strptime(end_date, date_format)
        if start > end:
            raise ValueError('Start date is bigger than end date.')
        self.start = start
        self.end = end


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


def download_files_to_directory(dirpath, downloadable_list, names_list):
    """Downloads files to directory.

    Parameters
    ----------
    dirpath : str
        Valid path to a directory.
    downloadable_list : str or list of str
        Valid urls pointing to downloadable files.
    names_list : str or list of str
        Names to give to the files in downloadable_list.

    Raises
    ------
    OSError
        Error is raised if the given dirpath is invalid according to os.path.isdir().

    """
    if not os.path.isdir(dirpath):
        raise OSError('Invalid Path.')
    if type(downloadable_list) is not list:
        downloadable_list = [downloadable_list]
    if type(names_list) is not list:
        names_list = [names_list]

    for url, name in zip(downloadable_list, names_list):
        print(f'Downloading file {name}')
        r = requests.get(url)
        with open(os.path.join(os.path.abspath(dirpath), name), 'wb') as fd:
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
    if start_datetime < datetime.datetime(2005, 6, 1, 0, 50) or end_datetime > datetime.datetime(2019, 12, 31, 23, 50):
        return False
    else:
        return True


def download_and_extract():
    """Downloads and extracts the DWD data.

    Checks if dwd_dl.config.config_initializer() was run. It runs it if that is False. Then generates a list of
    the files to be downloaded, asks if they should be downloaded, downloads them to a tmp directory and extracts them
    to the RADOLAN_PATH specified folder.

    """

    print('Checking if configuration function was run.')
    global DATE_RANGES_PATH
    if DATE_RANGES_PATH is None:
        print("It was not, I'm running it now!")
        config_initializer()

    download_list, file_names_list, sizes_list = radar_download_list_generator(
        date_ranges_path=DATE_RANGES_PATH
    )

    total_size = 0

    print('Computing total size.')
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
    """Cleans unused files from RADOLAN_PATH

    """
    global RADOLAN_PATH
    global DATE_RANGES_PATH

    assert config_was_run()

    listdir = os.listdir(os.path.abspath(RADOLAN_PATH))
    date_ranges = read_ranges(DATE_RANGES_PATH)

    for file in tqdm(listdir):
        if 'dwd---bin' not in file:
            continue
        file_used = False
        for date_range in date_ranges:
            if file in used_files(date_range.start, date_range.end):
                file_used = True
        if not file_used:
            os.remove(os.path.join(os.path.abspath(RADOLAN_PATH), file))


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
    for n in range(int((end_date - start_date).total_seconds()/3600) + end):
        yield start_date + datetime.timedelta(hours=n)


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


def config_initializer(config_fpath='.'):
    """Cheks if global variables are set and are viable. Otherwise it checks for a .cfg file. It creates one if it
    doesn't exist.

    Globals are set by reading them from the `RADOLAN.cfg` if available. Otherwise it resorts to default values.
    The `.cfg` file should have this format:
    `
    [DEFAULT]
    radolan_path = /path/to/base/Radolan
    date_ranges_path = /path/to/date/ranges
    start_date = YYYY-MM-DD HH:MM:SS
    end_date = YYYY-MM-DD HH:MM:SS
    `

    Where the dates are obviously substituted with valid ones.

    Parameters
    ----------
    config_fpath : str
        A valid path to the location of the `RADOLAN.cfg` file. (Defaults to pwd.)

    """

    globals_to_check = ['RADOLAN_PATH', 'DATE_RANGES_PATH', 'START_DATE', 'END_DATE']
    config_fname = 'RADOLAN.cfg'
    date_ranges_fname = 'DATE_RANGES.cfg'
    date_format = '%Y-%m-%d %H:%M:%S'

    config = configparser.ConfigParser()
    try:
        config.read(os.path.join(os.path.abspath(config_fpath), config_fname))
    except TypeError:
        raise SyntaxError('Wrong syntax in Radolan.cfg')

    for var in globals_to_check:
        if var not in config['DEFAULT']:
            raise ValueError(f'Missing variable in Radolan.cfg: {var}')
        else:
            globals()[var] = config['DEFAULT'][var]

    if not os.path.isdir(os.path.abspath(config['DEFAULT'][globals_to_check[0]])):
        raise OSError('Invalid RADOLAN_PATH')
    else:
        globals()['RADOLAN_PATH'] = os.path.abspath(config['DEFAULT'][globals_to_check[0]])

    print('Checking date ranges path.')
    check_date_ranges_path(config['DEFAULT'][globals_to_check[1]], date_ranges_fname)  # TODO: REFACTOR!!!
    global DATE_RANGES_PATH
    DATE_RANGES_PATH = os.path.join(os.path.abspath(config['DEFAULT'][globals_to_check[1]]), date_ranges_fname)

    print('Validating dates')
    for date in globals_to_check[2:]:
        try:
            globals()[date] = datetime.datetime.strptime(config['DEFAULT'][date], date_format)
        except ValueError:
            error_message = 'Wrong date format for {}. Expected format is: %Y-%m-%d %H:%M:%S.'.format(
                                date,
                                date,
                                str(globals()['DEFAULT_' + date])
                            )
            raise ValueError(error_message)

        if not (globals()['DEFAULT_START_DATE'] <= globals()[date] <= globals()['DEFAULT_END_DATE']):
            error_message = '{} {} should be between {} and {}.'.format(
                date,
                str(globals()[date]),
                str(globals()['DEFAULT_START_DATE']),
                str(globals()['DEFAULT_END_DATE']),
                str(globals()['DEFAULT_' + date])
            )
            raise ValueError(error_message)

    if globals()['START_DATE'] > globals()['END_DATE']:
        error_message = 'START_DATE {} is greater than END_DATE {}.'.format(
            str(globals()['START_DATE']),
            str(globals()['END_DATE'])
        )
        raise ValueError(error_message)

    # print('Writing config file.')
    # for var in globals_to_check:
    #     config['DEFAULT'][var] = str(globals()[var])
    #
    # with open(os.path.join(os.path.abspath(config_fpath), config_fname), 'w') as configfile:
    #     config.write(configfile)

    print('Setting global variable CONFIG_WAS_RUN.')
    global CONFIG_WAS_RUN
    CONFIG_WAS_RUN = True

    return DATE_RANGES_PATH


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


def binary_file_name(time_stamp):
    """Returns the file name of a DWD binary given a datetime.datetime timestamp.

    Parameters
    ----------
    time_stamp : datetime.datetime
        A valid timestamp.

    Returns
    -------
    str
        The name of the binary corresponding to the given timestamp. `raa01-rw_10000-yymmDDHHMM-dwd---bin`

    """
    return 'raa01-rw_10000-{}-dwd---bin'.format(time_stamp.strftime('%y%m%d%H%M'))


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
    return np.sqrt((np.subtract(a_tup, b_tup)**2).sum(axis=2))


def coords_finder(lat, lon, distances_output=False):
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
