"""
Methods intended for reduction of raw SANS data collected at the MacSANS laboratory at McMaster Nuclear Reactor.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

from math import sqrt
from numpy import linspace


def circular_binning(img, radius, center=(0, 0)):
    """
    Pixels that fall within a circle of radius r about a center point at (m1, m2):
    P = {(x,y) : (x-m1)^2 + (y-m2)^2 <= r^2}

    This function returns a list of tuples corresponding to these points.

    :param img: A 2D numpy array. This is usually 2D detector array data.
    :param radius: Radius of the circular bin about a center point.
    :param center: The index (row, col) of the circle center point.
    :return indices: A list of tuples containing the (row, col) indices of pixels that fall within the circle.
    """

    radius = float(radius)
    indices = []
    radius_squared = radius * radius

    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x-center[1]
            dy = y-center[0]
            distance_squared = float(dx * dx + dy * dy)

            if distance_squared <= radius_squared:
                indices.append((y, x))
    return indices


def ring_binning(outer, inner):
    """
    This function takes two lists of indices corresponding to pixels that fall within a circular bin. These are usually
    obtained from the circular_binning function. Using list comprehension, the indices from a small inner circle are
    subtracted from the indices of a larger outer circle to produce a list of indices that corresponds to all pixels
    that fall within the ring between the two.

    :param outer: A list of (row, col) tuples corresponding to pixel indices within a circle of radius r_outer about a
                  center point (y_center, x_center). r_outer must satisfy the condition r_outer > r_inner.
    :param inner: A list of (row, col) tuples corresponding to pixel indices with a circle of radius r_inner about a
                  center point (y_center, x_center). r_inner must satisfy the condition r_inner < r_outer.
    :return ring: A list of (row, col) tuples corresponding to pixels that fall within a ring bounded by r_outer and
                  r_inner.
    """

    ring = [x for x in outer if x not in inner]
    return ring


def get_intensity(img, indices):
    """
    This function takes a 2D array of counts and a list of (row, col) indices. It returns the average of counts found at
    points corresponding to the list of indices and returns an integer of intensity.

    :param img: A 2D array of counts, usually array data.
    :param indices: A list of (row, col) tuples corresponding to valid indices with img.
    :return intensity: The sum of all elements within img whose indices (row, col) can be found within the param
                       'indices'.
    """
    intensity = 0
    for y, row in enumerate(img):
        for x, val in enumerate(row):
            point = (y, x)
            if point in indices:
                intensity += int(val)
    intensity = intensity / len(indices)
    return intensity


def get_intensity_as_a_function_of_radius_in_pixels(img, center=(0, 0), n_bins=100):
    """
    Returns a list of radially averaged scattered intensities and their associated radial bins in pixels.

    :param img: A 2D array of image data.
    :param center: A tuple of the (row, col) index of the pixel at the center of the beam center or other point of
                   interest.
    :param n_bins: The number of bins used by radial binning methods.
    :return intensities: A list of radially averaged intensities.
    :return bins: A list defining the edges (in units of pixels) of radial bins used for averaging.
    """
    # Get the distance to the pixel farthest from the center point
    distance_to_farthest_pixel = 0.0
    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            distance = sqrt(dx * dx + dy * dy)
            if distance > distance_to_farthest_pixel:
                distance_to_farthest_pixel = distance
    # Make list of tuples where each element contains in inner and outer diameter of a ring
    bins = linspace(0.0, distance_to_farthest_pixel, num=n_bins)

    # Use bins as ring boundaries to calculate a list of intensities
    intensities = []
    for i, _ in enumerate(bins):
        if i == (len(bins) - 1):
            continue
        inner = circular_binning(img, bins[i], center)
        outer = circular_binning(img, bins[i+1], center)
        ring = ring_binning(outer, inner)
        intensity = get_intensity(img, ring)
        intensities.append(intensity)
    return intensities, bins
