import math
import numpy as np
from ccns.reduction.scattering import *


def test__wide_angle_correction_factor():
    theta = 45.0
    transmission = 0.9
    expected_factor = 0.9538737412050747
    result_factor = _wide_angle_correction_factor(theta, transmission)
    assert math.isclose(result_factor, expected_factor, rel_tol=1e-8)


def test__get_radial_bin():
    img = np.zeros((10, 10))  # Example image
    center = (5, 5)  # Center point
    r0 = 3  # Radius of the annular bin
    dr = 2  # Width of the annular bin
    expected_indices = [(1, 5), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
                        (3, 8), (4, 2), (4, 3), (4, 7), (4, 8), (5, 1), (5, 2), (5, 3), (5, 7), (5, 8), (5, 9), (6, 2),
                        (6, 3), (6, 7), (6, 8), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (8, 3), (8, 4),
                        (8, 5), (8, 6), (8, 7), (9, 5)]
    result_indices = _get_radial_bin(img, center, r0, dr)
    assert result_indices == expected_indices


def test__pixel_solid_angle():
    distance = 480.0  # Sample-to-detector distance in cm
    pixel_dim = (0.7, 0.7)  # Pixel dimensions in cm
    expected_solid_angle = 2.126736111111111e-06
    result_solid_angle = _pixel_solid_angle(distance, pixel_dim)
    assert math.isclose(result_solid_angle, expected_solid_angle, rel_tol=1e-9)


def test__get_scattering_angle():
    pixel = (10, 10)  # Pixel coordinates (row, col)
    center = (5, 5)  # Center coordinates (row, col)
    l2 = 480.0  # Sample-source-to-detector distance in cm
    pixel_dim = (0.7, 0.7)  # Pixel dimensions in cm
    expected_angle = 0.010311608401501715
    result_angle = _get_scattering_angle(pixel, center, l2, pixel_dim)
    assert math.isclose(result_angle, expected_angle, rel_tol=1e-9)

