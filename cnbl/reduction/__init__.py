"""
This subpackage contains all methods necessary for SANS data reduction at the MacSANS laboratory at McMaster University.
Found here are functions which are intended to be called by users. Modules within this subpackage contain subroutines
called by these user functions and are not intended to be accessed directly by most users.

    author: Devin Burke

(c) Copyright 2022, McMaster University
"""

import math
import numpy
from .scattering import *
from .gaussiandq import *

__all__ = ['solid_angle_correction',
           'scale_to_absolute_intensity',
           'estimate_incoherent_scattering',
           'get_beam_stop_factor',
           'get_scattered_intensity',
           'get_q_statistics',
           'resolution_function']


def solid_angle_correction(img, l2, center, pixel_dim=(0.7, 0.7)):
    """
    Perform solid angle correction of SANS data for planar detectors. This is usually the first step in data correction.
    The value of each pixel is multiplied by a geometric correction factor cos^3(theta). This function operates on the
    2D array you pass to it and returns None.

    :param img:            A 2D array of detector pixel values.
    :param l2:             sample-source-to-detector distance
    :param center:         A tuple designating the (row, col) indices of the center pixel of the radial bin
    :param pixel_dim:      A tuple (y, x) containing the y and x size of the pixel. (0.7, 0.7) by default.
    :return
    """

    distance = l2
    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            radius = math.sqrt((dx * pixel_dim[1]) ** 2 + (dy * pixel_dim[0]) ** 2)
            theta = math.atan(radius/distance)
            img[y][x] *= math.cos(theta) ** 3
    return


def scale_to_absolute_intensity(measured_img, empty_img,
                                sample_transmission,
                                sample_thickness,
                                sample_detector_distance,
                                illuminated_sample_area,
                                detector_efficiency,
                                counting_time,
                                monitor_counts,
                                normalize_time=False):
    """
    Scale the SANS data to form the macroscopic scattering cross section (units of cm^-1). The result is the absolute
    intensity.

    :param measured_img: 2D SANS data to be scaled.
    :param empty_img: SANS data of an empty beam.
    :param sample_transmission: Neutron transmission factor T of sample
    :param sample_thickness: Thickness of sample
    :param sample_detector_distance: sample-aperture-to-detector distance
    :param illuminated_sample_area: Sample area illuminated by neutrons
    :param detector_efficiency: The efficiency of the detector
    :param counting_time: The effective counting time.
    :param monitor_counts: The integral counts of the monitor or area detector.
    :param normalize_time: Set to True to normalize the counting time to 10^8 monitor counts.
    :return: 2D SANS data scaled to absolute intensity
    """

    pixel_solid_angle = _pixel_solid_angle(sample_detector_distance)

    if measured_img.shape != empty_img.shape:
        raise Exception("The shape of the measured scattering intensity with the sample %s must match the shape of the "
                        "empty beam measure %s" % (measured_img.shape, empty_img.shape))
    if normalize_time:
        # The effective counting time should be re-normalized to give 10^8 monitor counts
        counting_time *= (100000000.0 / monitor_counts)
    for y, row in enumerate(measured_img):
        for x, val in enumerate(row):
            empty_beam_transmission = (empty_img[y][x] * illuminated_sample_area * detector_efficiency
                                       * counting_time)
            m = (empty_beam_transmission * sample_transmission * sample_thickness * pixel_solid_angle)
            measured_img[y][x] *= 1 / m
    return


def estimate_incoherent_scattering(l2, sample_transmission, shape=(147, 147)):
    """
    This function estimates the incoherent scattering assuming that incoherent scattering dominates.

    :param l2:                      sample-aperture-to-detector distance.
    :param sample_transmission:     Neutron transmission factor T of sample.
    :param shape:                   Shape of the returned incoherent scattering array. (147, 147) by default.
    :return incoherent_scattering:  A 2D array of the estimated incoherent scattering.
    """

    distance = l2
    incoherent_scattering = numpy.zeros(shape, order='F')
    for y, row in enumerate(incoherent_scattering):
        for x, val in enumerate(row):
            incoherent_scattering[y][x] = 1 / (4*numpy.pi*distance) * (1 - sample_transmission) / sample_transmission
    return incoherent_scattering


def get_beam_stop_factor(r0, dr, b_s):
    """
    If the beam-stop is centered on the beam, this method return '0' for some point at distance r from the beam center
    across the detector plane if that point lies in the shadow of the beam-stop. It returns '1' otherwise.

    :param r0:           The radius of the annular bin.
    :param dr:           The full width of the annular bin.
    :param b_s:          The beam-stop radius.
    :return bool:
    """
    inner_radius = r0 - dr/2
    if b_s < inner_radius:
        return 1
    else:
        return 0


def get_scattered_intensity(abs_img, center, r0, dr, transmission, thickness, l2, pixel_dim=(0.7, 0.7)):
    """
    Calculates the mean and standard deviation of the scattering intensity within an annulus of radius r0 with width dr.

    :param abs_img:      2D SANS data scaled to absolute intensity
    :param center:       A tuple designating the (row, col) indices of the center pixel of the radial bin
    :param r0:           Radius of the annulus centered on the beam.
    :param dr:           Width of the annulus
    :param transmission: The neutron transmission factor of the sample.
    :param thickness:    The sample thickness
    :param l2:           source-aperture-to-detector distance
    :param pixel_dim:    A tuple (y, x) containing the y and x size of the pixel. (0.7, 0.7) by default.
    :return (mean, std): A tuple containing the mean and standard deviation of the scattered intensity within the
                         annulus.
    """
    distance = l2
    radial_bin = _get_radial_bin(abs_img, center, r0, dr)
    intensities = numpy.empty(0)
    for pixel in radial_bin:
        row = pixel[0]
        col = pixel[1]
        theta = _get_scattering_angle(pixel, center, distance, pixel_dim)
        wac = _wide_angle_correction_factor(theta, transmission, thickness)
        intensities = numpy.append(intensities, abs_img[row][col] * wac)
    if len(intensities) == 0:
        return 0, 0
    else:
        return numpy.mean(intensities), numpy.std(intensities)


def get_q_statistics(r_0, d_r, b_s, wl, wl_spread, sigma_d, l_1, l_2, s_1, s_2):
    """
    Returns the mean momentum transfer and its variance for an annulus with radius r_0 and width d_r after beam-stop and
    second order corrections.

    :param r_0: Radius of the annulus
    :param d_r: Width of the annulus
    :param b_s: Radius of the beam-stop.
    :param wl: The mean neutron wavelength.
    :param wl_spread: The standard deviation of the neutron wavelength distribution expressed as a fraction of its mean.
    :param sigma_d: Standard deviation of detector intrinsic spatial resolution
    :param l_1: source-aperture-to-sample-aperture distance
    :param l_2: sample-aperture-to-detector distance
    :param s_1: source aperture radius
    :param s_2: sample aperture radius
    :return q_mean: The mean momentum transfer in the radial bin
    :return v_q: The variance of the resolution function in q
    """

    # Get the variance contributions from the beam and from gravity
    v_rb = _vrb(l_1, l_2, s_1, s_2)
    v_rg = _vrg(wl, wl_spread, l_1, l_2)

    # Get the variance contribution from the detector and correct for the presence of the beamstop
    v_rd = _vrd(sigma_d, d_r)
    v_rds = _beam_stop_correction(v_rd, r_0, b_s, sigma_d)

    # Calculate the total variance in distance
    v_rs = _vr(v_rb, v_rds, v_rg)

    # Calculate the fractional correction to the mean distance from the beam-stop and second order effects
    f_r = _fr(v_rd, r_0, b_s, sigma_d)[0]
    r_mean, v_r = _second_order_size_effects(f_r, r_0, v_rs)

    # Convert to momentum transfer space
    # q = _get_q(r, l_2, wl)
    q_mean = _get_q(r_mean, l_2, wl)

    # Get the variance of the resolution function
    v_q = _q_variance(q_mean, v_r, r_0, wl_spread)

    # Get the value of the resolution function R at point (q, q_mean)
    # resolution = 1 / math.sqrt(2*numpy.pi*v_q) * math.exp(-1 * (q - q_mean)**2 / (2*v_q))

    return q_mean, v_q


def resolution_function(q, mean_q, v_q):
    """
    Returns the un-normalized value of the resolution function at point (q, mean_q). Values of this function should be
    normalized such at the integral of R(q, q0) over all q is equal to one.

    :param q: momentum transfer at some point r within an annulus of radius r_0 and width d_r.
    :param mean_q: mean momentum transfer within an annulus.
    :param v_q: variance of the momentum transfer within the annulus
    :return: The value of R(q, mean_q)
    """
    return 1 / math.sqrt(2*numpy.pi*v_q) * math.exp(-1 * (q - mean_q)**2 / (2*v_q))
