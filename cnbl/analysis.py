"""
Methods for analysis of reduced data using xarrays.

    author: Devin Burke

(c) Copyright 2023, McMaster University
"""

import xarray


def get_xarray(dat):
    x = xarray.DataArray(list(dat.values()), dims=("data_name", "Q_0"),
                         coords={'data_name': list(dat.keys()),
                                 'Q_0': dat['Q_0']})
    return x
