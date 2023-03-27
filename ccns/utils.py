"""
Utility functions useful when performing data analysis, visualization, and reduction. Intended for use at the MacSANS
laboratory at McMaster University

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

from scipy.integrate import cumtrapz
import numpy as np

def print_impact_matrix(dat, title=None):
    import matplotlib.pyplot as plt
    h = dat
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Impact Matrix')
    plt.imshow(h)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()

def normalize(array):
    integral = cumtrapz(array).sum()
    normalized_array = [float(i)/float(integral) for i in array]
    return normalized_array

def get_projections(a):
    y = np.mean(a, axis=0)
    x = np.mean(a, axis=1)
    return y,x
