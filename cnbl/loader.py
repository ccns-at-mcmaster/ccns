"""
Methods to load a nexus format raw SANS data file and return useful dictionaries of experimental data and metadata.
Intended for use at the MacSANS laboratory at McMaster University.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

import h5py
from os import path

nxsas_dict = {
    'root': '/',
    'entry': '/entry',
    'title': '/entry/title',
    'start_time': '/entry/start_time',
    'end_time': '/entry/end_time',
    'instrument': '/entry/instrument',
    'instrument_name': '/entry/instrument/name',
    'source': '/entry/instrument/source',
    'probe': '/entry/instrument/source/probe',
    'wavelength': '/entry/instrument/monochromator/wavelength',
    'wavelength_spread': '/entry/instrument/monochromator/wavelength_spread',
    'collimation_length': '/entry/instrument/collimator/geometry/shape/size',
    'sdd': '/entry/instrument/detector/distance',
    'x_pixel_size': '/entry/instrument/detector/x_pixel_size',
    'y_pixel_size': '/entry/instrument/detector/y_pixel_size',
    'beam_center_x': '/entry/instrument/detector/beam_center_x',
    'beam_center_y': '/entry/instrument/detector/beam_center_y',
    'sample_name': '/entry/sample',
    'monitor_mode': '/entry/monitor/mode',
    'monitor_preset': '/entry/monitor/preset',
    'monitor_integral': '/entry/monitor/integral',
    'data': '/entry/data/data',
    'metadata': '/entry/metadata'
}


def get_nexus_file(filepath, mode):
    """
    This method takes a filepath and open mode for a nexus file and returns the file.

    :param filepath: Filepath to the nexus data file
    :param mode: Open mode of the file, must be one of ['r', 'r+', 'w', 'w-', 'x', 'a']. See h5py.File for more info.
    :return nexus: An h5py nexus file object.
    """
    if path.exists(filepath):
        possible_modes = ['r', 'r+', 'w', 'w-', 'x', 'a']
        if mode in possible_modes:
            nexus = h5py.File(filepath, mode)
        else:
            # noinspection PyTypeChecker
            raise Exception('The file must be opened by specifying '
                            'one of the following modes {}'.format(possible_modes))
    else:
        warning = 'A nexus file at {} does not exist'.format(filepath)
        # noinspection PyTypeChecker
        raise Exception(warning)
    return nexus


if __name__ == "__main__":
    raw_filepath = "C:\\Users\\dburk\\Desktop\\working\\user\\newrocktest0001.raw"
    file = get_nexus_file(raw_filepath, 'r+')
    print(file['/entry/monitor/integral'][()])
