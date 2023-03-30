"""
Utility functions useful when performing data analysis, visualization, and reduction. Intended for use at the MacSANS
laboratory at McMaster University

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""
from scipy.integrate import cumtrapz
import numpy as np


def print_impact_matrix(dat, title='Impact Matrix', cmap=None, norm=None, aspect=None, interpolation=None, alpha=None,
                        vmin=None, vmax=None, origin=None, extent=None, interpolation_stage=None, filternorm=True,
                        filterrad=4.0, resample=None, url=None, data=None, **kwargs):
    import matplotlib.pyplot as plt
    h = dat
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(title)

    plt.imshow(h, cmap=cmap, norm=norm, aspect=aspect, interpolation=interpolation, alpha=alpha, vmin=vmin, vmax=vmax,
               origin=origin, extent=extent, interpolation_stage=interpolation_stage, filternorm=filternorm,
               filterrad=filterrad, resample=resample, url=url, data=data, **kwargs)

    ax.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    plt.show()


def normalize(array):
    integral = cumtrapz(array).sum()
    normalized_array = [float(i)/float(integral) for i in array]
    return normalized_array


def get_projections(a):
    y = np.mean(a, axis=0)
    x = np.mean(a, axis=1)
    return y, x
