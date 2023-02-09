"""
Methods for analysis of reduced data using xarrays.

    author: Devin Burke

(c) Copyright 2023, McMaster University
"""
import xarray
import matplotlib.pyplot as plt
import numpy

__all__ = ['get_xarray',
           'get_standard_plot']


def get_xarray(dat):
    """
    Takes a python dict of reduced SANS data and return an xarray DataArray object labeled by names of the data series
    and Q. An xarray object is useful for fast analysis of data and is expected by our analysis functions.

    :param dat:
    :return:
    """
    vals = {'I': dat['I'], 'Idev': dat['Idev'], 'Q': dat['Q'], 'Qdev': dat['Qdev'], 'BS': dat['BS']}
    x = xarray.DataArray(list(vals.values()), dims=("name", "Q"), coords={'name': list(vals.keys()),
                                                                          'Q': vals['Q']})
    return x


def _get_guinier_plot(x, q_range=None):
    """
    Generates a matplotlib plot of Ln(I) vs Q^2 (i.e. a standard Guinier plot) from the passed xarray.DataArray
    containing reduced SANS data.

    :param x: A DataArray that must contain a data series labeled 'I' and a coordinate labeled 'Q'.
    :param q_range: A python slice(min, max) object where min and max describe the range of Q values you want to include
                    in the analysis.
    :return line: A matplotlib.lines.Line2D object.
    :return xr: A new DataArray labeled with 'Ln(I)' and 'Q2' of the Guinier Plot data.
    """
    xr = x.copy()

    title = 'Guinier Plot'
    x_label = r'${\rm Q^2}\ {\rm (\AA^{-2})}$'
    y_label = r'${\rm Ln(I)}$'

    if isinstance(q_range, slice):
        xr = numpy.log(xr.sel(name='I', Q=q_range))
    if q_range is None:
        xr = numpy.log(xr.sel(name='I'))

    q2 = numpy.array(xr['Q'])
    q2 = numpy.power(q2, 2)
    xr = xr.assign_coords({'Q': q2})
    xr = xr.assign_coords({'name': 'Ln(I)'})
    xr = xr.rename({'Q': 'Q2'})
    line = xr.plot()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return line, xr


def _get_porod_plot(x, q_range=None):
    """
    Generates a matplotlib plot of Log(I(Q)-B) vs Log(Q) (i.e. a standard Porod plot) from the passed xarray.DataArray
    containing reduced SANS data.

    :param x: A DataArray that must contain a data series labeled 'I' and a coordinate labeled 'Q'.
    :param q_range: A python slice(min, max) object where min and max describe the range of Q values you want to include
                    in the analysis.
    :return line: A matplotlib.lines.Line2D object.
    :return xr: A new DataArray labeled with 'Log(I(Q)-B)' and 'Log(Q)' of the Porod Plot data.
    """
    xr = x.copy()
    title = 'Porod Plot'
    x_label = 'Log(Q)'
    y_label = 'Log(I(Q)-B)'

    if isinstance(q_range, slice):
        xr = numpy.log10(xr.sel(name='I', Q=q_range))
    if q_range is None:
        xr = numpy.log10(xr.sel(name='I'))

    log_q = numpy.array(xr['Q'])
    log_q = numpy.log10(log_q)
    xr = xr.assign_coords({'Q': log_q})
    xr = xr.assign_coords({'name': 'Log(I)'})
    xr = xr.rename({'Q': 'Log(Q)'})
    line = xr.plot()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return line, xr


def _get_zimm_plot(x, q_range=None):
    xr = x.copy()

    title = 'Zimm Plot'
    x_label = r'${\rm Q^2}\ {\rm (\AA^{-2})}$'
    y_label = '1/I(Q)'

    if isinstance(q_range, slice):
        xr = 1 / xr.sel(name='I', Q=q_range)
    if q_range is None:
        xr = 1 / xr.sel(name='I')

    q2 = numpy.array(xr['Q'])
    q2 = numpy.power(q2, 2)
    xr = xr.assign_coords({'Q': q2})
    xr = xr.assign_coords({'name': '1/I'})
    xr = xr.rename({'Q': 'Q2'})
    line = xr.plot()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return line, xr
    return


def get_standard_plot(name=None, data_array=None, q_range=None):
    """
    This method takes a xarray and calls a method to generate one of a list of standard plots.

    :param name: The name of the standard plot. Must be one of ['guinier', 'porod', 'zimm', 'kratky'].
    :param data_array: An xarray.DataArray object containing reduced SANS data.
    :param q_range: A python slice(min, max) object where min and max describe the range of Q values you want to include
                    in the analysis.
    :return:
    """
    xr = data_array
    standard_plots = ['guinier', 'porod', 'zimm', 'kratky']
    if name.lower() not in standard_plots:
        print("You must specify which of the standard plots {} you would like to generate.".format(standard_plots))
    if name.lower() == 'guinier':
        return _get_guinier_plot(xr, q_range)
    if name.lower() == 'porod':
        return _get_porod_plot(xr, q_range)
    if name.lower() == 'zimm':
        return _get_zimm_plot(xr, q_range)
    return
