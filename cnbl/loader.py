"""
Methods to load a nexus format raw SANS data file and return useful dictionaries of experimental data and metadata.
Intended for use at the MacSANS laboratory at McMaster University.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

import h5py
from os import path

nxsas_dict = {
    'title': '/entry/title',
    'start_time': '/entry/start_time',
    'end_time': '/entry/end_time',
    'instrument_name': '/entry/instrument/name',
    'source_type': '/entry/instrument/source/source_type',
    'source_name': '/entry/instrument/source/source_name',
    'source_probe': '/entry/instrument/source/source_probe',
    'monochromator_wavelength': '/entry/instrument/monochromator/wavelength',
    'monochromator_wavelength_spread': '/entry/instrument/monochromator/wavelength_spread',
    'collimation_length': '/entry/instrument/collimator/geometry/shape/size',
    'sdd': '/entry/instrument/detector/distance',
    'x_pixel_size': '/entry/instrument/detector/x_pixel_size',
    'y_pixel_size': '/entry/instrument/detector/y_pixel_size',
    'beam_center_x': '/entry/instrument/detector/beam_center_x',
    'beam_center_y': '/entry/instrument/detector/beam_center_y',
    'x_resolution': '/entry/instrument/detector/x_resolution',
    'y_resolution': '/entry/instrument/detector/y_resolution',
    'sample_name': '/entry/sample/sample_name',
    'monitor_mode': '/entry/monitor/mode',
    'monitor_preset': '/entry/monitor/preset',
    'monitor_integral': '/entry/monitor/integral',
    'data': '/entry/data/data',
    'metadata_detector_height': '/entry/instrument/metadata/detector_height',
    'metadata_counting_time': '/entry/instrument/metadata/counting_time',
    'metadata_sample_transmission': '/entry/instrument/metadata/sample_transmission',
    'metadata_sample_thickness': '/entry/instrument/metadata/sample_thickness'
}


def get_nexus_file(filepath, mode='r'):
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


def read_sans_raw(nexus):
    """
    This method takes a nexus file, extracts data and metadata from the file using the paths specified in nxsas_dict
    and returns a dictionary of the experimental data.

    :param nexus: A MacSANS raw nexus file. This should contain HDF5 databases at each path specified in nxsas_dict.
    :return data_dict: A dictionary with key/value pairs of experimental data and metadata.
    """
    data_dict = {}
    for key in nxsas_dict.keys():
        try:
            data_dict[key] = nexus[nxsas_dict[key]][()]
        except KeyError:
            data_dict[key] = None
    return data_dict
