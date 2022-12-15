"""
Methods intended for reduction of raw SANS data collected at the MacSANS laboratory at McMaster Nuclear Reactor.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

from math import sqrt, cos, atan
from numpy import linspace


def get_radial_bin(img, outer_radius, inner_radius, center):
    """
    This function is used for radial binning of SANS data. It returns a list of (row, col) indices corresponding to
    pixels that fall within a ring of arbitrary thickness about an arbitrary center point. This function defines two
    circles about a center point of different radii. It then checks each pixel in an image and calculates its distance
    to the center point. If that distance falls within between the radii of the two circles, it is appended to the list
    of indices.

        :param img:          A 2D numpy array. This is usually 2D detector array data.
        :param outer_radius: The outer radius (in pixels) of the ring defining a radial bin.
        :param inner_radius: The inner radius (in pixels) of the ring defining a radial bin. It must satisfy the
                             condition inner_radius < outer_radius
        :param center:       A tuple designating the (row, col) indices of the center pixel of the radial bin
        :return ring:        A list of (row, col) tuples which are indices corresponding to pixels that fall within a
                             ring bounded by outer_radius and inner_radius.
    """
    if inner_radius >= outer_radius:
        raise Exception('The inner radius of the radial binning ring must be less than the outer radius.')
    inner_radius = float(inner_radius)
    outer_radius = float(outer_radius)
    indices = []
    outer_radius_squared = outer_radius * outer_radius
    inner_radius_squared = inner_radius * inner_radius

    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            distance_squared = float(dx * dx + dy * dy)

            if inner_radius_squared <= distance_squared <= outer_radius_squared:
                indices.append((y, x))
    return indices


def get_average_intensity(img, indices):
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

    :param img:          A 2D array of image data.
    :param center:       A tuple of the (row, col) index of the pixel at the center of the beam center or other point of
                         interest.
    :param n_bins:       The number of bins used by radial binning methods.
    :return intensities: A list of radially averaged intensities.
    :return bins:        A list defining the edges (in units of pixels) of radial bins used for averaging.
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
    # Define the bounds of radial bins out to the
    bins = linspace(0.0, distance_to_farthest_pixel, num=n_bins)

    # Use bins as ring boundaries to calculate a list of intensities
    intensities = []
    for i, _ in enumerate(bins):
        if i == (len(bins) - 1):
            continue
        radial_bin = get_radial_bin(img, bins[i+1], bins[i], center)
        average_intensity = get_average_intensity(img, radial_bin)
        intensities.append(average_intensity)
    return intensities, bins


def solid_angle_correction(img, center, distance, calibration=0.7):
    """
    Perform solid angle correction of SANS data for planar detectors. This is usually the first step in data correction.
    The value of each pixel is multiplied by a geometric correction factor cos^3(theta). This function operates on the
    2D array you pass to it and returns None.

    :param img:            A 2D array of detector pixel values
    :param center:         A (row, col) tuple containing indices corresponding to the pixel closest to the center of the
                           beam.
    :param distance:       Sample-to-detector distance in cm.
    :param calibration:    The real size of the detector pixel in cm. This should be 0.7 cm for the Mirrotron 2D
                           detector.
    :return
    """

    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            radius = sqrt(dx * dx + dy * dy)
            radius *= calibration
            theta = atan(radius/distance)
            img[y][x] *= cos(theta) * cos(theta) * cos(theta)
    return
