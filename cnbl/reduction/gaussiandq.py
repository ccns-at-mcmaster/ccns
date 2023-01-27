"""
Methods intended for reduction of raw SANS data collected at the MacSANS laboratory at McMaster Nuclear Reactor.
These methods are used for the gaussian method of calculating the q resolution.

    Author: Devin Burke

(c) Copyright 2022, McMaster University

Method from:
Barker, J. G., and J. S. Pedersen. "Instrumental smearing effects in radially symmetric small-angle neutron scattering
by numerical and analytical methods." Journal of applied crystallography 28.2 (1995): 105-114.
"""

import math
import numpy
import scipy.constants as const
from scipy.special import gammainc


def _vrb(l1, l2, s1, s2):
    """
    The variance produced by the beam divergence for circular apertures.

    Seeger, P.A. (1980). Nucl. Instrum. Methods, 178, 157-161
    Mildner, D.F.R. & Carpenter,J.M. (1984). J. Appl. Cryst. 17, 249-256

    :param l1: source-aperture-to-sample-aperture distance
    :param l2: sample-aperture-to-detector distance
    :param s1: source aperture radius
    :param s2: sample aperture radius
    :return vrb: beam variance in distance
    """
    lp = 1 / ((1 / l1) + (1 / l2))
    vrb = ((s1**2*l2**2) / (4*l1**2)) + ((s2**2*l2**2) / (4*lp**2))
    return vrb


def _vrd(sig_d, dr):
    """
    Variance produced by the detector

    :param sig_d: standard deviation of detector intrinsic spatial resolution
    :param dr: width of the radial average annulus
    :return Vrd: Variance produced by the detector.
    """
    Vrd = sig_d ** 2 + (dr ** 2 / 12)
    return Vrd


def _vrg(wl, wl_spread, l1, l2):
    """
    Variance produced by deflection of the beam due to gravity

    :param wl: neutron mean wavelength
    :param wl_spread: Relative spread of the neutron wavelength. sigma_wavelength / wl. Usually 0.1 - 0.2.
    :param l1: source-aperture-to-sample-aperture distance
    :param l2: sample-aperture-to-detector distance
    :return Vrg: variance produced by deflection of the beam due to gravity.
    """
    v_neutron = const.h / (const.m_n * wl) * 1E12
    Yg = (const.g * 100 / (2 * v_neutron ** 2)) * l2 * (l1 + l2)
    Vrg = 2 * Yg ** 2 * wl_spread ** 2
    return Vrg


def _vr(vrb, vrd, vrg):
    """
    Total variance in distance with contributions from the beam, the detector, and gravity.

    :param vrb: Variance produced by the beam divergence for circular apertures.
    :param vrd: Variance produced by the detector
    :param vrg: Variance produced by deflection of the beam due to gravity
    :return vr: Total variance in distance
    """
    vr = vrb + vrd + vrg
    return vr


def _q_variance(q0, vr, r0, wl_spread):
    """
    Variance in the momentum transfer q.

    :param q0: Nominal value of the momentum transfer.
    :param vr: Variance in distance between the points of intersection of the incident and scattered rays with the
               detector plane.
    :param r0: Nominal scattering distance across the detector plane. The center of the radially averaged bin.
    :param wl_spread: Relative spread of the neutron wavelength. sigma_wavelength / wl. Usually 0.1 - 0.2.
    :return Vq: Variance in the momentum transfer q.
    """
    try:
        x = vr / (r0 * r0)
        y = wl_spread ** 2
        Vq = q0 * q0 * (x + y)
    except ZeroDivisionError:
        return 0
    return Vq


def _fs(vrd, r0, bs):
    """
    The reduction in detector efficiency for a pixel caused by the beam-stop shadow.
    This effect may also be corrected for by normalizing the data to the scattering from water with the beam stop
    in the same location for both the sample- and water-scattering runs.

    :param vrd: variance produced by the detector
    :param bs: radius of the beam-stop shadow
    :param r0: Nominal scattering distance across the detector plane. The center of the radially averaged bin.
    :return fs: Fractional reduction in detector efficiency for a pixel caused by the beam-stop shadow.
    :return delta_r: A value useful in calculating fr and fv.
    """
    delta_r = (r0 - bs) / math.sqrt(2 * vrd)
    fs = 0.5 * (1 + math.erf(delta_r))
    return fs, delta_r


def _fr(vrd, r0, bs, sig_d):
    """
    Fractional shift in mean distance caused by the beam-stop shadow. The beam-stop shadow preferentially screens
    scattering events of small scattering distance r, which shifts the mean value to a larger distance and also reduces
    the variance. Division by the scattering from water does not correct for this effect.

    :param vrd: variance produced by the detector
    :param r0: Nominal scattering distance across the detector plane. The center of the radially averaged bin.
    :param bs: radius of the beam-stop shadow
    :param sig_d: standard deviation of detector intrinsic spatial resolution
    :return fr: Fractional shift in mean distance caused by the beam-stop shadow. fr(r0) = mean distance / r0
    :return fs: Fractional reduction in detector efficiency for a pixel caused by the beam-stop shadow.
    :return delta_r: A value useful in calculating fr and fv.
    """
    fs, delta_r = _fs(vrd, r0, bs)
    x = sig_d * numpy.exp(-1 * delta_r ** 2)
    y = r0 * fs * math.sqrt(2 * numpy.pi)
    try:
        fr = 1 + x/y
    except ZeroDivisionError:
        fr = 1
    return fr, fs, delta_r


def _fv(vrd, r0, bs, sig_d):
    """
    As r0 -> 0, mean distance -> bs. This is the corresponding fractional shift in the detector component of
    the distance variance.

    :param vrd: variance produced by the detector.
    :param r0: Nominal scattering distance across the detector plane. The center of the radially averaged bin.
    :param bs: radius of the beam-stop shadow.
    :param sig_d: standard deviation of detector intrinsic spatial resolution.
    :return fv: Fractional shift in the detector component of the distance variance. fv = Vrds / Vrd
    """
    fr, fs, delta_r = _fr(vrd, r0, bs, sig_d)
    x = 1 / (fs * math.sqrt(numpy.pi)) * (1 - gammainc(3/2, delta_r**2))
    y = (r0 ** 2 / vrd) * ((fr - 1) ** 2)
    fv = x - y
    return fv


def _beam_stop_correction(v_rd, r_0, b_s, sig_d):
    """
    Corrects the variance produced by the detector for beam-stop effects.

    :param v_rd: variance produced by the detector.
    :param r_0: Nominal scattering distance across the detector plane. The center of the radially averaged bin.
    :param b_s: radius of the beam-stop shadow.
    :param sig_d: standard deviation of detector intrinsic spatial resolution.
    :return v_rds: beam-stop corrected variance in distance produced by the detector
    """
    fv = _fv(v_rd, r_0, b_s, sig_d)
    v_rds = fv * v_rd
    return v_rds


def _second_order_size_effects(f_r, r0, vrs):
    """
    Deviations between the mean distance r_mean and the nominal distance r0 occur for data elements near the beam from
    other second-order size effects.

    Mildner, D.F.R. & Carpenter,J.M. (1984). J. Appl. Cryst. 17, 249-256

    :param f_r: Fractional shift in mean distance caused by the beam-stop shadow. fr(r0) = mean distance / r0
    :param r0: Nominal scattering distance across the detector plane. The center of the radially averaged bin.
    :param vrs: The total variance after v_rd has been corrected for beam-stop effects to v_rds.
    :return:
    """
    rd = f_r * r0
    r_mean = rd + (vrs / (2 * rd))
    vr = vrs - (vrs**2 / (2 * rd**2))
    return r_mean, vr


def _get_q(r, l2, wl):
    """
    Get the magnitude of the momentum transfer Q for a given scattering distance across the detector face for a given
    neutron wavelength.

    :param r: Scattering distance across the detector plane relative to the beam center.
    :param l2: sample-aperture-to-detector distance
    :param wl: neutron mean wavelength
    :return q: magnitude of the momentum transfer
    """
    theta = math.atan(r / l2)
    q = 4 * numpy.pi / wl * math.sin(theta/2)
    return q


def get_q_statistics(r_0, d_r, b_s, wl, wl_spread, sigma_d, l_b, l_1, l_2, s_1, s_2):
    """
    Returns the mean momentum transfer and its variance for an annulus with radius r_0 and width d_r after beam-stop and
    second order corrections.

    :param r_0: Radius of the annulus
    :param d_r: Width of the annulus
    :param b_s: Radius of the beam-stop.
    :param wl: The mean neutron wavelength.
    :param wl_spread: The standard deviation of the neutron wavelength distribution expressed as a fraction of its mean.
    :param sigma_d: Standard deviation of detector intrinsic spatial resolution
    :param l_b: distance between the beam-stop and the detector plane.
    :param l_1: source-aperture-to-sample-aperture distance
    :param l_2: sample-aperture-to-detector distance
    :param s_1: source aperture radius
    :param s_2: sample aperture radius
    :return q_mean: The mean momentum transfer in the radial bin
    :return v_q: The variance of the resolution function in q
    """

    # Effective beam stop radius
    b_eff = b_s * l_2 / (l_2 - l_b)

    # Get the variance contributions from the beam and from gravity
    v_rb = _vrb(l_1, l_2, s_1, s_2)
    v_rg = _vrg(wl, wl_spread, l_1, l_2)

    # Get the variance contribution from the detector and correct for the presence of the beamstop
    v_rd = _vrd(sigma_d, d_r)
    v_rds = _beam_stop_correction(v_rd, r_0, b_eff, sigma_d)

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
