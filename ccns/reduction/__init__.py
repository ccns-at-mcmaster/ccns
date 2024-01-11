"""
This subpackage contains all methods necessary for SANS data reduction at the MacSANS laboratory at McMaster University.
Found here are functions which are intended to be called by users. Modules within this subpackage contain subroutines
called by these user functions and are not intended to be accessed directly by most users.

    author: Devin Burke

(c) Copyright 2023, McMaster University
"""

import math
import numpy
import datetime
from .scattering import *
from .gaussiandq import *

__all__ = ['solid_angle_correction',
           'scale_to_absolute_intensity',
           'estimate_incoherent_scattering',
           'get_beam_stop_factor',
           'get_scattered_intensity',
           'get_q_statistics',
           'resolution_function',
           'reduce_data',
           'rescale_with_empty_and_blocked_beams']


def solid_angle_correction(img, distance, center, pixel_dim=(0.7, 0.7)):
    """
    Perform solid angle correction of SANS data for planar detectors. This is usually the first step in data correction.
    The value of each pixel is multiplied by a geometric correction factor cos^3(theta). This function operates on the
    2D array you pass to it and returns None.

    :param img:            A 2D array of detector pixel values.
    :param distance:             sample-source-to-detector distance
    :param center:         A tuple designating the (row, col) indices of the center pixel of the radial bin
    :param pixel_dim:      A tuple (y, x) containing the y and x size of the pixel. (0.7, 0.7) by default.
    :return A 2D array corrected for planar geometry.
    """
    corrected_img = img.copy()
    for y, row in enumerate(corrected_img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            radius = math.sqrt((dx * pixel_dim[1]) ** 2 + (dy * pixel_dim[0]) ** 2)
            theta = math.atan(radius/distance)
            corrected_img[y][x] *= math.cos(theta) ** 3
    return corrected_img


def scale_to_absolute_intensity(img,
                                empty_img,
                                center,
                                sdd,
                                sample_transmission,
                                sample_thickness,
                                illuminated_sample_area,
                                detector_efficiency,
                                counting_time,
                                monitor_counts,
                                normalize_time=False,
                                pixel_dim=(0.7, 0.7)):
    """
    Scale the SANS data to form the macroscopic scattering cross section (units of cm^-1). The result is the absolute
    intensity.

    :param img: 2D SANS data to be scaled.
    :param empty_img: SANS data of an empty beam.
    :param center: A tuple designating the (row, col) indices of the center pixel of the radial bin
    :param sdd: sample-to-detector distance
    :param sample_transmission: Neutron transmission factor T of sample
    :param sample_thickness: Thickness of sample
    :param illuminated_sample_area: Sample area illuminated by neutrons
    :param detector_efficiency: The efficiency of the detector
    :param counting_time: The effective counting time.
    :param monitor_counts: The integral counts of the monitor or area detector.
    :param normalize_time: Set to True to normalize the counting time to 10^8 monitor counts.
    :param pixel_dim: A tuple (y, x) containing the y and x size of the pixel. (0.7, 0.7) by default.
    :return: 2D SANS data scaled to absolute intensity
    """
    scaled_img = numpy.copy(img)
    # This SDD is the distance from the sample to the detector
    pixel_solid_angle = _pixel_solid_angle(sdd)

    if scaled_img.shape != empty_img.shape:
        # noinspection PyTypeChecker
        raise Exception("The shape of the measured scattering intensity with the sample %s must match the shape of the "
                        "empty beam measure %s" % (scaled_img.shape, empty_img.shape))
    if normalize_time:
        # The effective counting time should be re-normalized to give 10^8 monitor counts
        counting_time *= (100000000.0 / monitor_counts)
    for y, row in enumerate(scaled_img):
        for x, val in enumerate(row):
            theta = _get_scattering_angle((y, x), center, sdd, pixel_dim)
            corrected_transmission = sample_transmission * _wide_angle_correction_factor(theta, sample_transmission)
            empty_beam_transmission = (empty_img[y][x] * illuminated_sample_area * detector_efficiency
                                       * counting_time)
            m = (empty_beam_transmission * corrected_transmission * sample_thickness * pixel_solid_angle)
            if m == 0.0:
                pass
            else:
                scaled_img[y][x] *= 1 / m
    return scaled_img


def estimate_incoherent_scattering(distance, sample_transmission, shape=(147, 147)):
    """
    This function estimates the incoherent scattering assuming that incoherent scattering dominates.

    :param distance:                sample-to-detector distance.
    :param sample_transmission:     Neutron transmission factor T of sample.
    :param shape:                   Shape of the returned incoherent scattering array. (147, 147) by default.
    :return incoherent_scattering:  A 2D array of the estimated incoherent scattering.
    """
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
    :return :            0 if the point is in the shadow of the beam-stop, 1 otherwise.
    """
    if all([isinstance(x, float) for x in [r0, dr, b_s]]):
        inner_radius = r0 - dr/2
        if b_s < inner_radius:
            return 1
        if b_s >= inner_radius:
            return 0
    else:
        raise Exception("The parameters r0, dr, and b_s must be floats.")


def get_scattered_intensity(abs_img, center, r0, dr):
    """
    Calculates the mean and standard deviation of the scattering intensity within an annulus of radius r0 with width dr.

    :param abs_img:      2D SANS data scaled to absolute intensity
    :param center:       A tuple designating the (row, col) indices of the center pixel of the radial bin
    :param r0:           Radius of the annulus centered on the beam.
    :param dr:           Width of the annulus
    :return (mean, std): A tuple containing the mean and standard deviation of the scattered intensity within the
                         annulus.
    """
    radial_bin = _get_radial_bin(abs_img, center, r0, dr)
    intensities = numpy.empty(0)
    for pixel in radial_bin:
        row = pixel[0]
        col = pixel[1]
        intensities = numpy.append(intensities, abs_img[row][col])
    # noinspection PyTypeChecker
    if len(intensities) == 0:
        return 0, 0
    else:
        return numpy.mean(intensities), numpy.std(intensities)


def get_q_statistics(r_0, d_r, b_s, wl, wl_spread, sigma_d, sdd, l_1, l_2, s_1, s_2):
    """
    Returns the mean momentum transfer and its variance for an annulus with radius r_0 and width d_r after beam-stop and
    second order corrections.

    :param r_0: Radius of the annulus
    :param d_r: Width of the annulus
    :param b_s: Radius of the beam-stop.
    :param wl: The mean neutron wavelength.
    :param wl_spread: The standard deviation of the neutron wavelength distribution expressed as a fraction of its mean.
    :param sigma_d: Standard deviation of detector intrinsic spatial resolution
    :param sdd: sample-to-detector distance
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
    if b_s > 0.0:
        v_rds = _beam_stop_correction(v_rd, r_0, b_s, sigma_d)
    else:
        v_rds = v_rd

    # Calculate the total variance in distance
    v_rs = _vr(v_rb, v_rds, v_rg)

    # Calculate the fractional correction to the mean distance from the beam-stop and second order effects
    if b_s > 0.0:
        f_r = _fr(v_rd, r_0, b_s, sigma_d)[0]
        r_mean, v_r = _second_order_size_effects(f_r, r_0, v_rs)
    else:
        r_mean, v_r = r_0, v_rs

    # Convert to momentum transfer space
    # q = _get_q(r, l_2, wl)
    q_mean = _get_q(r_mean, sdd, wl)

    # Get the variance of the resolution function
    v_q = _q_variance(q_mean, v_r, r_0, wl_spread)
    q_std = math.sqrt(v_q)

    # Get the value of the resolution function R at point (q, q_mean)
    # resolution = 1 / math.sqrt(2*numpy.pi*v_q) * math.exp(-1 * (q - q_mean)**2 / (2*v_q))

    return q_mean, q_std


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


def reduce_data(sans_data,
                annulus_width,
                center,
                beamstop_radius,
                neutron_wavelength,
                wavelength_spread,
                detector_resolution,
                sdd,
                l1,
                l2,
                s1,
                s2,
                pixel_dim=(0.7, 0.7),
                precision=6):
    """
    Returns a dictionary of reduced SANS data. The reduced data can be accessed with dict[key] where key is a string
    that can be 'Q', 'Q_variance', 'scattered_intensity', 'scattered_intensity_std', or 'BS'.

    :param sans_data: 2D SANS data
    :param annulus_width: Desired radial annulus width for radial averaging.
    :param center: A (row, col) tuple containing the pixel indices of the pixel closest to the beam center
    :param beamstop_radius: The radius of the beam-stop
    :param neutron_wavelength: The mean neutron wavelength
    :param wavelength_spread: The relative error in the mean neutron wavelength expressed as a decimal.
    :param detector_resolution: The standard deviation of the intrinsic detector spatial resolution.
    :param sdd: sample_to_detector distance
    :param l1: source aperture (S1) to sample aperture (S2) distance
    :param l2: sample aperture (S1) to detector distance
    :param s1: Radius of the source aperture
    :param s2: Radius of the sample aperture
    :param pixel_dim: A tuple (y, x) containing the y and x size of the pixel. (0.7, 0.7) by default.
    :param precision: The desired precision of Q statistics. Defaults to 6 decimal places.
    :return reduced data: A python dictionary with keys corresponding to reduced data. The value of each key is a numpy
                          array of the data. The reduced data can be accessed with dict[key] where key is a string that
                          can be 'Q', 'Q_variance', 'scattered_intensity', 'scattered_intensity_std', or 'BS'.
    """
    d_vars = [annulus_width, beamstop_radius, neutron_wavelength, wavelength_spread, detector_resolution, sdd, l1, l2, s1, s2]
    for i, x in d_vars:
        if not isinstance(x, float):
            d_vars[i] = float(x)

    if not all(isinstance(x, tuple) for x in [center, pixel_dim]):
        raise TypeError('center and pixel_dim must be tuples.')
    dr = annulus_width
    bs = beamstop_radius
    wl = neutron_wavelength
    wl_spread = wavelength_spread
    sigma_d = detector_resolution

    # Generate list of annular radii
    detector_axis_length = sans_data.shape[0] * pixel_dim[0]
    n_bins = int(detector_axis_length / dr)
    radii = numpy.linspace(0, detector_axis_length, n_bins)

    # noinspection PyTypeChecker
    reduced_data = {'Q': numpy.empty(0),
                    'I': numpy.empty(0),
                    'Idev': numpy.empty(0),
                    'Qdev': numpy.empty(0),
                    'ShadowFactor': numpy.empty(0, dtype=int),
                    'Q_0': numpy.empty(0),
                    'reduction_timestamp': datetime.datetime.now().isoformat()}
    for r0 in radii:
        if r0 <= dr:
            continue
        q, v_q = get_q_statistics(r0, dr, bs, wl, wl_spread, sigma_d, sdd, l1, l2, s1, s2)
        reduced_data['Q'] = numpy.append(reduced_data['Q'], round(q, precision))
        reduced_data['Qdev'] = numpy.append(reduced_data['Qdev'], round(v_q, precision))

        intensity, intensity_std = get_scattered_intensity(sans_data, center, r0, dr)
        reduced_data['I'] = numpy.append(reduced_data['I'], round(intensity, 0))
        reduced_data['Idev'] = numpy.append(reduced_data['Idev'], round(intensity_std, 0))

        bs_factor = get_beam_stop_factor(r0, dr, bs)
        reduced_data['ShadowFactor'] = numpy.append(reduced_data['ShadowFactor'], bs_factor)
        q_0 = _get_q(r0, l2, wl)
        reduced_data['Q_0'] = numpy.append(reduced_data['Q_0'], round(q_0, precision))
    return reduced_data


def rescale_with_empty_and_blocked_beams(sample_and_cell,
                                         beam_blocked,
                                         empty_cell,
                                         transmission_sample_and_cell,
                                         transmission_cell):
    """
    Uses the scattering matrices from the sample and cell together, an empty cell, and a blocked beam to rescale the
    scattering data. The returned matrix is the scattering from just the sample. This does not normalize the matrices
    so the user should make sure each matrix is collected at the same neutron flux and for the same counting time.
    Otherwise, the user should normalize the matrices first.

    :param sample_and_cell: A 2D numpy array. Scattering from both the sample and cell
    :param beam_blocked: Scattering with the beam block
    :param empty_cell: Scattering from just the empty_cell
    :param transmission_sample_and_cell: Neutron transmission factor of the sample and cell
    :param transmission_cell: Neutron transmission factor of the empty cell.
    :return sample_scattering: A 2D numpy array of the scattering from the sample alone.
    """

    # Sample run
    j = numpy.subtract(sample_and_cell, beam_blocked)
    j = numpy.divide(j, transmission_sample_and_cell)
    # Empty cell run
    k = numpy.subtract(empty_cell, beam_blocked)
    k = numpy.divide(k, transmission_cell)

    sample_scattering = numpy.subtract(j, k)
    return sample_scattering
