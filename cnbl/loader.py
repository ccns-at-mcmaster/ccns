"""
Methods to load a nexus format raw SANS data file and return useful dictionaries of experimental data and metadata.
Intended for use at the MacSANS laboratory at McMaster University.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

import h5py
from os import path


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
