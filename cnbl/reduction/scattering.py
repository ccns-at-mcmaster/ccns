"""
Methods intended for reduction of raw SANS data collected at the MacSANS laboratory at McMaster Nuclear Reactor.
These methods are used to calculate scattering vector quantities and absolute intensity.

    Author: Devin Burke

(c) Copyright 2022, McMaster University

Method from:
Barker, J. G., and J. S. Pedersen. "Instrumental smearing effects in radially symmetric small-angle neutron scattering
by numerical and analytical methods." Journal of applied crystallography 28.2 (1995): 105-114.
"""

import math
import numpy


def _wide_angle_correction_factor(theta):
    """
    Returns the wide-angle correction factor of intensity for a given scattering angle. This function should be called
    during radial averaging.

    :param theta: The scattering angle in degrees.
    :return:
    """
    return 1/math.cos(theta) - 1


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
    inner_radius = r0 - dr/2
    outer_radius = r0 + dr/2

    if inner_radius >= outer_radius:
        raise Exception('The inner radius of the radial binning ring must be less than the outer radius.')
    inner_radius = inner_radius
    outer_radius = outer_radius
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


def solid_angle_correction(img, l2, center, pixel_size):
    """
    Perform solid angle correction of SANS data for planar detectors. This is usually the first step in data correction.
    The value of each pixel is multiplied by a geometric correction factor cos^3(theta). This function operates on the
    2D array you pass to it and returns None.

    :param img:            A 2D array of detector pixel values.
    :param l2:             sample-source-to-detector distance
    :param center:         A tuple designating the (row, col) indices of the center pixel of the radial bin
    :param pixel_size      The linear size of a pixel along the x or y-axis (if equal).
    :return
    """

    distance = l2
    calibration = pixel_size

    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            radius = math.sqrt(dx * dx + dy * dy)
            radius *= calibration
            theta = math.atan(radius/distance)
            img[y][x] *= math.cos(theta) * math.cos(theta) * math.cos(theta)
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

    :param l2: sample-aperture-to-detector distance.
    :param sample_transmission: Neutron transmission factor T of sample.
    :param shape: Shape of the returned incoherent scattering array. (147, 147) by default.
    :return incoherent_scattering: A 2D array of the estimated incoherent scattering.
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


def get_scattered_intensity(abs_img, center, r0, dr):
    """
    Calculates the mean and standard deviation of the scattering intensity within an annulus of radius r0 with width dr.

    :param abs_img: 2D SANS data scaled to absolute intensity
    :param center:  A tuple designating the (row, col) indices of the center pixel of the radial bin
    :param r0: Radius of the annulus centered on the beam.
    :param dr: Width of the annulus
    :return (mean, std): A tuple containing the mean and standard deviation of the scattered intensity within the
                         annulus.
    """
    radial_bin = _get_radial_bin(abs_img, center, r0, dr)

    intensities = numpy.empty(1)
    for pixel in radial_bin:
        row = pixel[0]
        col = pixel[1]
        numpy.append(intensities, abs_img[row][col])
    intensities = numpy.array(intensities)
    mean = numpy.mean(intensities)
    std = numpy.std(intensities)
    return mean, std
