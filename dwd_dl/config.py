"""Module for checking requirements and downloading the radar data.

"""

import os


def radar_download_list_generator(start_year=2005, start_month=1, end_year=2019, end_month=12):
    base_url = 'https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/historical/bin/'
    download_list = []
    for year in range(start_year, end_year+1):
        for month in range(1, 13):
            if not (year == start_year and month < start_month) and not (year == end_year and month > end_month):
                download_list.append(base_url + f'{year}/' + 'RW{}{:02d}.tar.gz'.format(year, month))

    return download_list


def cfg_list(path):
    with open(path, 'r') as f:
        radar_files = f.readlines()
    return radar_files


def make_cfg(path, **kwargs):
    with open(path, 'w+') as f:
        f.writelines(radar_download_list_generator(**kwargs)) #TODO: Doesn't add \n
    return True


def cfg_handler(path=os.path.abspath('.')):
    if not os.path.isfile(os.path.join(path, 'RADOLAN.cfg')):
        make_cfg(os.path.join(path, 'RADOLAN.cfg'))

    DOWNLOAD_LIST = cfg_list(os.path.join(path, 'RADOLAN.cfg'))


