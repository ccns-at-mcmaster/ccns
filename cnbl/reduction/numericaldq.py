"""
Methods intended for reduction of raw SANS data collected at the MacSANS laboratory at McMaster Nuclear Reactor.
These methods are used for the numerical method of calculated the q resolution.

    Author: Devin Burke

(c) Copyright 2022, McMaster University

Method from:
Barker, J. G., and J. S. Pedersen. "Instrumental smearing effects in radially symmetric small-angle neutron scattering
by numerical and analytical methods." Journal of applied crystallography 28.2 (1995): 105-114.
"""

from scipy.integrate import quad, cumtrapz
from sympy.functions.elementary.exponential import exp as symexp
from sympy.functions.special.bessel import besseli
from sympy import N
import math
import numpy


def _pixel_response_function(r_d, sig_d):
    pixel_response = 1 / math.sqrt(2 * math.pi * sig_d**2)
    pixel_response *= math.exp(-1 * r_d**2 / (2 * sig_d**2))
    return pixel_response


def __circle_of_pixels_argument(r, r0, sig_d):
    f0 = (r / (sig_d ** 2))
    a = -1 * (r ** 2 + r0 ** 2) / (2 * sig_d ** 2)
    b = r * r0 / (sig_d ** 2)
    num = N((symexp(a) * besseli(0, b)))
    f0 *= num
    return f0


def _annulus_of_pixels_response_function(r, r0, sig_d, dr):
    arg = __circle_of_pixels_argument
    response_rdca = quad(lambda z: (r0+z)*arg(r, r0+z, sig_d), -0.5*dr, 0.5*dr)
    if nan_check(response_rdca):
        raise Exception(r, r0, sig_d, dr)
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
        al = (math.pi * d1**2 / 2) - (x_star * math.sqrt(d1**2 - x_star**2)) - d1**2 * math.asin(x_star / d1)
        ar = (math.pi * d2**2 / 2) -\
             (r - x_star) * math.sqrt((d2**2 - (r - x_star)**2)) - d2**2 * math.asin((r - x_star) / d2)
        a_e = (al + ar) / (math.pi * d ** 2)
        # a_e = 4 * (al + ar) / (pi * d ** 2)

    return a_e


def _rb_function(r, r0, psi):
    rb = math.sqrt(r**2 + r0**2 - 2 * r0 * r * math.cos(psi))
    return rb


def _beam_resolution_function(r, r0, d1, d2):
    a2 = 0.5 * (d1+d2)
    z = (r0**2 + r**2 - a2**2)/(2*r0*r)
    if -1 <= z <= 1:
        psi_max = math.acos(z)
    else:
        return 0
    res_function = quad(lambda psi: r * _beam_profile_function(_rb_function(r, r0, psi), d1, d2), 0, psi_max)
    if nan_check(res_function):
        raise Exception(r, r0, d1, d2)
    return res_function[0]


def nan_check(tup):
    for x in tup:
        if not x or numpy.isnan(x):
            return True
    return False


def __resolution_function_arg(r, r0, d1, d2, dr, sig_d, u):
    Rrb = _beam_profile_function
    Rdca = _annulus_of_pixels_response_function
    rb = _rb_function
    a2 = 0.5 * (d1 + d2)
    psi_max = math.acos(((r + u) ** 2 + r ** 2 - a2 ** 2) / (2 * (r + u) * r))
    z = quad(lambda psi: r*Rrb(rb(r, r0, psi), d1, d2)*(r+u)*Rdca(r+u, r0, sig_d, dr)*math.cos(psi), 0, psi_max)
    if nan_check(z):
        raise Exception(r, r0, d1, d2, dr, sig_d, u)
    return z[0]


def _simple_detector_resolution_function(r, r0, d1, d2, dr, bs, sig_d, pixel_size=0.7):
    a2 = 0.5 * (d1 + d2)
    # rdp radial detection limit for the pixel.
    rdp = math.sqrt(2 * (pixel_size/2) ** 2)
    if r0 + dr < bs:
        return 0.0
    u_min = max([-a2, r - r0 - rdp, bs-r])
    u_max = min([a2, r - r0 + rdp])
    # if u_max < u_min then the pixel is fully shadowed by the beam stop
    if u_max < u_min:
        return 0.0
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

    r_0 = 100
    func = []
    ordinate = []
    radii = numpy.linspace(r_0 - (d_r/2), r_0+(d_r/2), 100)

    for radius in radii:
        print(radius)
        if radius == 0.0:
            func.append(0.0)
            continue
        val = _simple_detector_resolution_function(radius, r_0, d_1, d_2, d_r, b_s, sigma_d, pixel)
        func.append(val)
        ordinate.append(radius)
    integral = cumtrapz(func).sum()
    if integral == 0.0:
        norm = [0.0] * len(ordinate)
    else:
        norm = [float(k)/integral for k in func]
    import matplotlib.pyplot as plt
    x_axis = ordinate
    y_axis = norm
    plt.plot(x_axis, y_axis)
    plt.title('r0='+str(r_0))
    plt.show()
