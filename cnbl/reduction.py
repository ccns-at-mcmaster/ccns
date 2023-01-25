"""
Methods intended for reduction of raw SANS data collected at the MacSANS laboratory at McMaster Nuclear Reactor.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

from math import sqrt, cos, exp, erf, sin, atan, asin, acos
from numpy import linspace, zeros, pi
import scipy.constants as const
from scipy.special import gammainc, iv
from scipy.integrate import quad

# Set precision
prec = 6

def _wide_angle_correction_factor(theta):
    """
    Returns the wide-angle correction factor of intensity for a given scattering angle. This function should be called
    during radial averaging.

    :param theta: The scattering angle in degrees.
    :return:
    """
    return 1/cos(theta) - 1


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
            distance = sqrt(dx * dx + dy * dy)

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
                radius = sqrt(dx * dx + dy * dy)
                radius *= calibration
                scattering_angle = atan(radius / distance)
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
            radius = sqrt(dx * dx + dy * dy)
            radius *= calibration
            theta = atan(radius/distance)
            img[y][x] *= cos(theta) * cos(theta) * cos(theta)
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
    incoherent_scattering = zeros(shape, order='F')
    for y, row in enumerate(incoherent_scattering):
        for x, val in enumerate(row):
            incoherent_scattering[y][x] = 1 / (4 * pi * distance) * (1 - sample_transmission) / sample_transmission
    return incoherent_scattering


def _radial_variance_beam_gaussian(L1, L2, S1, S2):
    Lp = 1 / ((1 / L1) + (1 / L2))
    Vrb = (S1 * S1 * L2 * L2 / (4 * Lp * Lp))
    return Vrb


def _radial_variance_detector_gaussian(sigma_d, delta_r):
    Vrd = sigma_d ** 2 + (delta_r ** 2 / 12)
    return Vrd


def _radial_variance_gravity_gaussian(wavelength, wavelength_spread, L1, L2):
    v_neutron = const.h / (const.m_n * wavelength) * 1E12
    Yg = (const.g * 100 / (2 * v_neutron ** 2)) * L2 * (L1 + L2)
    Vrg = 2 * Yg ** 2 * wavelength_spread ** 2
    return Vrg


def _q_variance_gaussian(q, Vr, r, Vw, w):
    try:
        x = Vr / (r * r)
        y = Vw / (w * w)
        Vq = q * q * (x + y)
    except ZeroDivisionError:
        return 0
    return Vq


def get_q(w, theta):
    q = 4 * pi / w * sin(theta/2)
    return q


def _reduction_in_pixel_efficiency_caused_by_beam_stop_shadow(r, Bs, Vrd):
    delta_r = (r - Bs) / sqrt(2 * Vrd)
    fs = 0.5 * (1 + erf(delta_r))
    return fs, delta_r


def _fractional_shift_in_mean_distance_caused_by_beam_stop_shadow(sigma_d, delta, r, fs):
    x = sigma_d * exp(-1 * delta ** 2)
    y = r * fs * sqrt(2 * pi)
    try:
        fr = 1 + x/y
    except ZeroDivisionError:
        return 1
    return fr


def _fractional_shift_in_radial_variance_detector(fs, delta, r, Vrd, fr):
    x = 1 / (fs * sqrt(pi)) * (1-gammainc(3/2, delta**2))
    y = (r ** 2 / Vrd) * ((fr - 1) ** 2)
    fv = x - y
    return fv


def beam_stop_correction(r, b_s, sigma_d, v_rd, v_rb, v_rg):
    fs, delta_r = _reduction_in_pixel_efficiency_caused_by_beam_stop_shadow(r, b_s, v_rd)
    fr = _fractional_shift_in_mean_distance_caused_by_beam_stop_shadow(sigma_d, delta_r, r, fs)
    fv = _fractional_shift_in_radial_variance_detector(fs, delta_r, r, v_rd, fr)
    v_rds = fv * v_rd
    v_rs = v_rb + v_rds + v_rg
    return v_rs, fr


def second_order_size_effects(fr, r, vrs):
    rd = fr * r
    corrected_r = rd + (vrs / (2 * rd))
    corrected_vr = vrs - (vrs * vrs / (2 * rd * rd))
    return corrected_r, corrected_vr


def _response_function(v_q, q, mean_q):
    response = 1 / sqrt(2 * pi * v_q) * exp((-1 * (q-mean_q) ** 2) / (2 * v_q))
    return response


def _pixel_response_function(rd, sig_d):
    pixel_response = 1 / sqrt(2 * pi * sig_d**2)
    pixel_response *= exp(-1 * rd**2 / (2 * sig_d**2))
    return pixel_response


def __circle_of_pixels_argument(r, r0, sig_d):
    f0 = (r / sig_d ** 2)
    f0 *= exp(-1 * (r ** 2 + r0 ** 2) / (2 * sig_d ** 2))
    # print(r, r0, sig_d)
    f0 *= iv(0, r * r0 / (sig_d ** 2))
    return f0


def _circle_of_pixels_response_function(r, r0, sig_d, dr):
    arg = __circle_of_pixels_argument
    response_rdca = quad(lambda z: (r0+z)*arg(r, r0+z, sig_d), -0.5*dr, 0.5*dr)
    return response_rdca[0]


def _beam_profile_function(r, d1, d2):
    d = min([d1, d2])
    x_star = (r**2 + d1**2 - d2**2) / (2 * r)
    # a1 = 0.5 * abs(d1-d2)
    # a2 = 0.5 * (d1+d2)

    # The commented lines are limits specified by Barker 1995. The 1/2 term in A1 and A2 when added to the limit
    # produces a complex value of the sqrt term in AL. The argument of the square root is positive so long as
    # |D1 - D2| <= r <= (D1 + D2). These should be the limits.
    # The argument of the sqrt term in AL is D1^2-[(r^2+D1^2-D2^2)/(2r)]^2.
    # Solving the expression (D1^2-[(r^2+D1^2-D2^2)/(2r)]^2) >= 0
    # Yields |D1-D2|<=r<=(D1+D2)
    # If you investigate the limit 0.5*|D1-D2| <= r <= 0.5*(D1+D2)
    # If A1 <= r <= (2A1 = |D1-D2|) the sqrt term is complex.
    # When calculating A_e, the paper says the expression is 4(AL+AR)/(pi*D^2). I removed the factor of 4
    # so that A_e = 1 at r = |D1-D2|
    # I suspect that the algebra treats D1 and D2 as radii instead of diameter in places.
    a_e = 0
    # if r < a1:
    if r < abs(d1-d2):
        a_e = 1
    # if r > a2:
    if r > d1 + d2:
        a_e = 0
    # if a1 <= r <= a2:
    if abs(d1 - d2) <= r <= d1+d2:
        al = (pi * d1**2 / 2) - (x_star * sqrt(d1**2 - x_star**2)) - d1**2 * asin(x_star / d1)
        ar = (pi * d2**2 / 2) - (r - x_star) * sqrt((d2**2 - (r - x_star)**2)) - d2**2 * asin((r - x_star) / d2)
        a_e = (al + ar) / (pi * d ** 2)
        # a_e = 4 * (al + ar) / (pi * d ** 2)

    return a_e


def _rb_function(r, r0, psi):
    rb = sqrt(r**2 + r0**2 - 2 * r0 * r * cos(psi))
    return rb


def _beam_resolution_function(r, r0, d1, d2):
    a2 = 0.5 * (d1+d2)
    z = (r0**2 + r**2 - a2**2)/(2*r0*r)
    if -1 <= z <= 1:
        psi_max = acos(z)
    else:
        return 0
    res_function = quad(lambda psi: r * _beam_profile_function(_rb_function(r, r0, psi), d1, d2), 0, psi_max)[0]
    return res_function[0]


def nan_check(tup):
    for x in tup:
        if x:
            return True
    return False


def __resolution_function_arg(r, r0, d1, d2, dr, sig_d, u):
    Rrb = _beam_profile_function
    Rdca = _circle_of_pixels_response_function
    rb = _rb_function
    a2 = 0.5 * (d1 + d2)
    print(r, u, a2)
    val = ((r + u) ** 2 + r ** 2 - a2 ** 2) / (2 * (r + u) * r)
    if val < -1 or val > 1:
        print(r, u, a2, val)
    psi_max = acos(val)
    z = quad(lambda psi: r*Rrb(rb(r, r0, psi), d1, d2)*(r+u)*Rdca(r+u, r0, sig_d, dr)*cos(psi), 0, psi_max)
    if nan_check(z):
        raise Exception(r, r0, d1, d2, dr, sig_d, u)
    return z[0]


def _simple_detector_resolution_function(r, r0, d1, d2, dr, bs, sig_d, pixel_size = 0.7):
    a2 = 0.5 * (d1 + d2)
    # rdp radial detection limit for the pixel. What is this?
    # rdp = sqrt(r ** 2 + r0 ** 2 - 2 * r * r0 * cos(pi))
    rdp = pixel_size/2
    u_min = max([-a2, r - r0 - rdp, bs - r])
    u_max = min([a2, r - r0 + rdp])
    print('u_min:', u_min)
    print(-a2, r-r0-rdp, bs-r)
    print('u_max:', u_max)
    print(a2, r-r0+rdp)
    arg = __resolution_function_arg
    z = quad(lambda u: arg(r, r0, d1, d2, dr, sig_d, u), u_min, u_max)
    if nan_check(z):
        raise Exception(r, r0, d1, d2, dr, sig_d, u_min, u_max)
    return z[0]


if __name__ == "__main__":
    """
    Example smearing calculation from Barker 1995.
    """
    w = wavelength = 5.0
    dw = wavelength_spread = 0.15
    sigma_d = detector_res_sd = 0.425
    d_r = annulus_width = 0.5
    b_s = beamstop_radius = 2.5
    l_b = beamstop_distance = 15
    s_1 = slit_one = 1.1
    s_2 = slit_two = 0.6
    l_1 = source_to_sample = 1630
    l_2 = sample_to_detector = 1530
    d_1 = 2 * s_1 * l_2 / l_1
    d_2 = 2 * s_2 * (l_1 + l_2) / l_1
    # Effective beam stop radius
    b_s = b_s * l_2 / (l_2 - l_b)
    pixel = 0.7

    r_0 = 10
    distances = linspace(r_0 - d_r, r_0 + d_r, 100)
    func = []
    for radius in distances:
        val = _simple_detector_resolution_function(radius, r_0, d_1, d_2, d_r, b_s, sigma_d, pixel)
        func.append(val)
    print(func)