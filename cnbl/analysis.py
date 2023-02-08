"""
Methods for analysis of reduced data using xarrays.

    author: Devin Burke

(c) Copyright 2023, McMaster University
"""

import xarray

__all__ = ['get_xarray']

def truncate_beam_shadow(dat):
    """
    Takes a dict containing reduced SANS data and deletes each element along the axis where the beamstop shadowfactor is
    0.

    :param dat:
    :return:
    """
    return

def get_xarray(dat):
    """


    :param dat:
    :return:
    """
    vals = {'I': dat['I'], 'Idev': dat['Idev'], 'Q': dat['Q'], 'Qdev': dat['Qdev'], 'ShadowFactor': dat['ShadowFactor']}
    x = xarray.DataArray(list(vals.values()), dims=("Value", "Q"), coords={'Value': list(vals.keys()),
                                                                           'Q': vals['Q']})
    return x
