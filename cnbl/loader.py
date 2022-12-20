"""
Contains methods to load a HDF5 nexus file. Intended for use on raw nexus files generated at the
MacSANS laboratory at McMaster University.

    author: Devin Burke

(c) Copyright 2022, McMaster University
"""

import h5py

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


def read_sans_raw(file):
    return


if __name__ == '__main__':
    filepath = "C:\\Users\\dburk\\OneDrive\\Desktop\\working\\"
    filename = "rocktest0000.raw"

    nexus = h5py.File(filepath+filename, 'r')
