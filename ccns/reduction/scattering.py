"""
Methods intended for reduction of raw SANS data collected at the MacSANS laboratory at McMaster Nuclear Reactor.
These methods are used to calculate scattering vector quantities and absolute intensity.

    Author: Devin Burke

(c) Copyright 2023, McMaster University
"""

import math

__all__ = ['_wide_angle_correction_factor',
           '_get_radial_bin',
           '_pixel_solid_angle',
           '_get_scattering_angle']


def _wide_angle_correction_factor(theta, transmission):
    """
    Returns the wide-angle correction factor of transmission for a given scattering angle.
    Calculated adapted from Hammouda's The SANS Toolbox.

    :param theta: The scattering angle in degrees.
    :param transmission: neutron transmission factor for the sample
    :return factor: The wide angle correction factor for the scattered intensity
    """
    a = (1 / math.cos(theta)) - 1
    if theta == 0.0 or transmission >= 1.0:
        return 1
    T = transmission
    correction = (math.pow(T, a) - 1) / (a * math.log(T))
    return correction


def _get_radial_bin(img, center, r0, dr):
    """
    This function is used for radial binning of SANS data. It returns a list of (row, col) indices corresponding to
    pixels that fall within a ring of arbitrary thickness about an arbitrary center point. This function defines two
    circles about a center point of different radii. It then checks each pixel in an image and calculates its distance
    to the center point. If that distance falls within between the radii of the two circles, it is appended to the list
    of indices.

        :param img:          Any 2D numpy array with the same shape as the 2D SANS data.
        :param center:       A tuple designating the (row, col) indices of the center pixel of the radial bin
        :param r0:           The radius of the annular bin.
        :param dr:           The full width of the annular bin.
        :return indices:     A list of (row, col) tuples which are indices corresponding to pixels that fall within a
                             ring bounded by outer_radius and inner_radius.
    """
    try:
        _ = iter(center)
    except TypeError as te:
        # noinspection PyTypeChecker
        print(center, 'must be iterable')
        raise te

    if dr <= 0:
        # noinspection PyTypeChecker
        raise Exception('The annulus width must be a number greater than zero.')

    inner_radius = r0 - dr/2
    outer_radius = r0 + dr/2

    indices = []

    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            distance = math.sqrt(dx * dx + dy * dy)

            if inner_radius <= distance <= outer_radius:
                indices.append((y, x))
    return indices


def _pixel_solid_angle(distance, pixel_dim=(0.7, 0.7)):
    """
    Returns the solid angle subtended by a detector pixel orthogonal to and centered on the beam axis.

    :param distance: The sample-to-detector distance in cm.
    :param pixel_dim: A tuple (x, y) containing the x and y size of the pixel
    :return solid_angle: The solid angle subtended by the chosen pixel.
    """

    pixel_area = pixel_dim[0] * pixel_dim[1]
    solid_angle = pixel_area / (distance * distance)
    return solid_angle


def _get_scattering_angle(pixel, center, l2, pixel_dim=(0.7, 0.7)):
    """
    Returns the scattering angle subtended by the distance between the center of an arbitrary pixel and the beam center.

    :param pixel: A tuple designating the (row, col) indices of a pixel
    :param center: A tuple designating the (row, col) indices of the center pixel of the beam center
    :param l2: sample-source-to-detector distance
    :param pixel_dim: A tuple (y, x) containing the y and x size of the pixel. (0.7, 0.7) by default.
    :return: The scattering angle
    """
    dx = pixel[1] - center[1]
    dy = pixel[0] - center[0]
    r = math.sqrt((dx * pixel_dim[1])**2 + (dy * pixel_dim[0])**2)
    return math.atan(r/l2)
