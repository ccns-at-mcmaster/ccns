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
    fig = plt.figure()

    ax = fig.add_subplot()
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Impact Matrix')
    plt.imshow(h)
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
    return y,x
