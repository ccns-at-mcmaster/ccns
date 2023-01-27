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


def _radial_variance_beam_gaussian(l1, l2, s1, s2):
    """
    The variance produced by the beam divergence for circular apertures.

    Seeger, P.A. (1980). Nucl. Instrum. Methods, 178, 157-161
    Mildner, D.F.R. & Carpenter,J.M. (1984). J. Appl. Cryst. 17, 249-256

    :param l1: source-aperture-to-sample-aperture distance
    :param l2: sample-aperture-to-detector distance
    :param s1: source aperture radius
    :param s2: sample aperture radius
    :return Vrb: beam variance
    """
    lp = 1 / ((1 / l1) + (1 / l2))
    Vrb = ((s1**2*l2**2) / (4*l1**2)) + ((s2**2*l2**2) / (4*lp**2))
    return Vrb


def _radial_variance_detector_gaussian(sig_d, delta_r):
    Vrd = sig_d ** 2 + (delta_r ** 2 / 12)
    return Vrd


def _radial_variance_gravity_gaussian(wl, wl_spread, l1, l2):
    v_neutron = const.h / (const.m_n * wl) * 1E12
    Yg = (const.g * 100 / (2 * v_neutron ** 2)) * l2 * (l1 + l2)
    Vrg = 2 * Yg ** 2 * wl_spread ** 2
    return Vrg


def _q_variance_gaussian(q0, vr, r0, wl_spread):
    try:
        x = vr / (r0 * r0)
        y = wl_spread ** 2
        Vq = q0 * q0 * (x + y)
    except ZeroDivisionError:
        return 0
    return Vq


def get_q(wl, theta):
    q = 4 * numpy.pi / wl * math.sin(theta/2)
    return q


def _reduction_in_pixel_efficiency_caused_by_beam_stop_shadow(r, bs, vrd):
    delta_r = (r - bs) / math.sqrt(2 * vrd)
    fs = 0.5 * (1 + math.erf(delta_r))
    return fs, delta_r


def _fractional_shift_in_mean_distance_caused_by_beam_stop_shadow(sig_d, delta, r, fs):
    x = sig_d * numpy.exp(-1 * delta ** 2)
    y = r * fs * math.sqrt(2 * numpy.pi)
    try:
        fr = 1 + x/y
    except ZeroDivisionError:
        return 1
    return fr


def _fractional_shift_in_radial_variance_detector(fs, delta, r, v_rd, fr):
    x = 1 / (fs * math.sqrt(numpy.pi)) * (1-gammainc(3/2, delta**2))
    y = (r ** 2 / v_rd) * ((fr - 1) ** 2)
    fv = x - y
    return fv


def beam_stop_correction(r, bs, sig_d, v_rd, v_rb, v_rg):
    fs, delta_r = _reduction_in_pixel_efficiency_caused_by_beam_stop_shadow(r, bs, v_rd)
    fr = _fractional_shift_in_mean_distance_caused_by_beam_stop_shadow(sig_d, delta_r, r, fs)
    fv = _fractional_shift_in_radial_variance_detector(fs, delta_r, r, v_rd, fr)
    v_rds = fv * v_rd
    v_rs = v_rb + v_rds + v_rg
    return v_rs, fr


def second_order_size_effects(f_r, r, vrs):
    rd = f_r * r
    corrected_r = rd + (vrs / (2 * rd))
    corrected_vr = vrs - (vrs * vrs / (2 * rd * rd))
    return corrected_r, corrected_vr


def _response_function(v_q, q, mean_q):
    response = 1 / math.sqrt(2 * numpy.pi * v_q) * numpy.exp((-1 * (q-mean_q) ** 2) / (2 * v_q))
    return response
