"""
Methods for analysis of reduced data using xarrays.

    author: Devin Burke

(c) Copyright 2023, McMaster University
"""
import xarray

__all__ = ['get_xarray']


def get_xarray(dat):
    """
    Takes a python dict of reduced SANS data and return an xarray DataArray object labeled by names of the data series
    and Q. An xarray object is useful for fast analysis of data and is expected by our analysis functions.

    :param dat:
    :return:
    """
    vals = {'I': dat['I'], 'Idev': dat['Idev'], 'Q': dat['Q'], 'Qdev': dat['Qdev'], 'BS': dat['BS']}
    x = xarray.DataArray(list(vals.values()), dims=("Value", "Q"), coords={'name': list(vals.keys()),
                                                                           'Q': vals['Q']})
    return x
