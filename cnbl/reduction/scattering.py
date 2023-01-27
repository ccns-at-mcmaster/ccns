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


def _get_radial_bin(img, outer_radius, inner_radius, center):
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


def _get_average_intensity(img, indices, center, distance, calibration=0.7):
    """
    This function takes a 2D array of counts and a list of (row, col) indices. It returns the average of counts found at
    points corresponding to the list of indices and returns an integer of intensity.

    :param img:         A 2D array of counts, usually array data.
    :param indices:     A list of (row, col) tuples corresponding to valid indices with img.
    :param center:      A (row, col) tuple containing indices corresponding to the pixel closest to the center of the
                        beam.
    :param distance:    Sample-to-detector distance in cm.
    :param calibration: The real size of the detector pixel in cm. This should be 0.7 cm for the Mirrotron 2D
                        detector.
    :return intensity:  The sum of all elements within img whose indices (row, col) can be found within the param
                        'indices'.
    """

    intensity = 0
    for y, row in enumerate(img):
        for x, val in enumerate(row):
            point = (y, x)
            if point in indices:
                dx = x - center[1]
                dy = y - center[0]
                radius = math.sqrt(dx * dx + dy * dy)
                radius *= calibration
                scattering_angle = math.atan(radius / distance)
                intensity += val * _wide_angle_correction_factor(scattering_angle)
    intensity = intensity / len(indices)
    return intensity


def get_intensity_as_a_function_of_radius_in_pixels(img, data, n_bins=100):
    """
    Returns a list of radially averaged scattered intensities and their associated radial bins in pixels.

    :param img:          A 2D array of image data.
    :param data:         The dictionary of experiment data and metadata.
    :param n_bins:       The number of bins used by radial binning methods.
    :return intensities: A list of radially averaged intensities.
    :return bins:        A list defining the edges (in units of pixels) of radial bins used for averaging.
    """

    distance = data['sdd'][0]
    center = (int(data['beam_center_x'][0]), int(data['beam_center_y'][0]))
    # Get the distance to the pixel farthest from the center point
    distance_to_farthest_pixel = 0.0
    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > distance_to_farthest_pixel:
                distance_to_farthest_pixel = distance
    # Define the bounds of radial bins out to the
    bins = numpy.linspace(0.0, distance_to_farthest_pixel, num=n_bins)

    # Use bins as ring boundaries to calculate a list of intensities
    intensities = []
    for i, _ in enumerate(bins):
        if i == (len(bins) - 1):
            continue
        radial_bin = _get_radial_bin(img, bins[i + 1], bins[i], center)
        average_intensity = _get_average_intensity(img, radial_bin, center, distance)
        intensities.append(average_intensity)
    return intensities, bins


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


def solid_angle_correction(img, data):
    """
    Perform solid angle correction of SANS data for planar detectors. This is usually the first step in data correction.
    The value of each pixel is multiplied by a geometric correction factor cos^3(theta). This function operates on the
    2D array you pass to it and returns None.

    :param img:            A 2D array of detector pixel values.
    :param data:           The dictionary of experiment data and metadata.
    :return
    """

    distance = data['sdd'][0]
    center = (int(data['beam_center_x'][0]), int(data['beam_center_y'][0]))
    calibration = data['x_pixel_size'][0]

    for y, row in enumerate(img):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            radius = math.sqrt(dx * dx + dy * dy)
            radius *= calibration
            theta = math.atan(radius/distance)
            img[y][x] *= math.cos(theta) * math.cos(theta) * math.cos(theta)
    return


def scale_to_absolute_intensity(measured_img, empty_img, data, normalize_time=False):
    """
    Scale the SANS data to form the macroscopic scattering cross section (units of cm^-1). The result is the absolute
    intensity.

    :param measured_img:            2D array of scattering measured with a sample.
    :param empty_img:               2D array of scattering measured with an empty beam.
    :param data:                    The dictionary of experiment data and metadata.
    :param normalize_time:          A boolean value. If true the counting time will be normalized to 10^8 monitor
                                    counts.
    :return scaled_img:             A 2D array of absolute intensity (units of cm^-1)
    """
    sample_transmission = data['metadata_sample_transmission'][0]
    sample_thickness = data['metadata_sample_thickness'][0]
    sample_detector_distance = data['sdd'][0]
    illuminated_sample_area = data['metadata_sample_area'][0]
    detector_efficiency = data['detector_efficiency'][0]
    counting_time = data['metadata_counting_time'][0]
    monitor_counts = int(data['monitor_integral'][0])

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
    return measured_img


def estimate_incoherent_scattering(data):
    """
    This function estimates the incoherent scattering assuming that incoherent scattering dominates.

    :param data:                   The dictionary of experiment data and metadata.
    :return incoherent_scattering: An estimate of the incoherent scattering of the sample. It is a 2D array with shape
                                   specified by the shape parameter.
    """

    distance = data['sdd'][0]
    shape = data['data'].shape
    sample_transmission = data['metadata_sample_transmission'][0]
    incoherent_scattering = numpy.zeros(shape, order='F')
    for y, row in enumerate(incoherent_scattering):
        for x, val in enumerate(row):
            incoherent_scattering[y][x] = 1 / (4*numpy.pi*distance) * (1 - sample_transmission) / sample_transmission
    return incoherent_scattering


def beam_stop_factor(r, b_s, l_b, l_2):
    """
    If the beam-stop is centered on the beam, this method return '1' for some point at distance r from the beam center
    across the detector plane if that point lies in the shadow of the beam-stop. It returns '0' otherwise.

    :param r: The distance between some point on the detector face and the beam center.
    :param b_s: The radius of the beam-stop.
    :param l_b: The distance between the beam-stop and the detector face.
    :param l_2: The sample-aperture-to-detector distance.
    :return bool:
    """
    b_eff = b_s * l_2 / (l_2 - l_b)
    if r <= b_eff:
        return 1
    else:
        return 0
