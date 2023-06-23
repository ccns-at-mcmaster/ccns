from ccns.reduction import *
import numpy as np
import math
import datetime


def test_solid_angle_correction():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    distance = 480.0
    center = (1, 1)
    pixel_dim = (0.7, 0.7)
    expected_img = np.array([[0, 1, 2],
                             [3, 5, 5],
                             [6, 7, 8]])
    result_img = solid_angle_correction(img, distance, center, pixel_dim)
    assert np.allclose(result_img, expected_img, rtol=1e-9)


def test_scale_to_absolute_intensity():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    empty_img = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]])
    center = (1, 1)
    sdd = 480.0
    sample_transmission = 0.8
    sample_thickness = 0.1
    illuminated_sample_area = 3.14
    detector_efficiency = 0.7
    counting_time = 1.0
    monitor_counts = 100000.0
    normalize_time = False
    pixel_dim = (0.7, 0.7)

    expected_img_false = np.array([[5348091, 5348090, 5348091],
                                   [5348090, 5348090, 5348090],
                                   [5348091, 5348090, 5348091]])
    result_img_false = scale_to_absolute_intensity(img, empty_img, center, sdd, sample_transmission, sample_thickness,
                                                   illuminated_sample_area, detector_efficiency, counting_time,
                                                   monitor_counts, normalize_time, pixel_dim)
    expected_img_true = np.array([[5348, 5348, 5348],
                                  [5348, 5348, 5348],
                                  [5348, 5348, 5348]])
    normalize_time = True
    result_img_true = scale_to_absolute_intensity(img, empty_img, center, sdd, sample_transmission, sample_thickness,
                                                  illuminated_sample_area, detector_efficiency, counting_time,
                                                  monitor_counts, normalize_time, pixel_dim)

    assert np.allclose(result_img_false, expected_img_false, rtol=1e-9)
    assert np.allclose(result_img_true, expected_img_true, rtol=1e-9)


def test_estimate_incoherent_scattering():
    distance = 480.0
    sample_transmission = 0.8
    shape = (3, 3)

    expected_incoherent_scattering = np.array([[4.14465998e-05, 4.14465998e-05, 4.14465998e-05],
                                               [4.14465998e-05, 4.14465998e-05, 4.14465998e-05],
                                               [4.14465998e-05, 4.14465998e-05, 4.14465998e-05]])
    result_incoherent_scattering = estimate_incoherent_scattering(distance, sample_transmission, shape)

    assert np.allclose(result_incoherent_scattering, expected_incoherent_scattering, rtol=1e-9)


def test_get_beam_stop_factor():
    r0 = 10.0
    dr = 2.0
    b_s = 5.0
    result1 = get_beam_stop_factor(r0, dr, b_s)
    r0 = 3.0
    result2 = get_beam_stop_factor(r0, dr, b_s)
    assert result1 == 1
    assert result2 == 0


def test_get_scattered_intensity():
    abs_img = [[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9]]
    center = (1, 1)
    r0 = 1.5
    dr = 1.0
    expected_mean = 0.5
    expected_std = 0.2738612787525831
    mean, std = get_scattered_intensity(abs_img, center, r0, dr)
    assert math.isclose(mean, expected_mean, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(std, expected_std, rel_tol=1e-9, abs_tol=1e-9)


def test_get_q_statistics():
    input_parameters = {
                        'r_0': 1.0,
                        'd_r': 0.1,
                        'b_s': 0.05,
                        'wl': 1.0,
                        'wl_spread': 0.1,
                        'sigma_d': 0.01,
                        'sdd': 5.0,
                        'l_1': 2.0,
                        'l_2': 3.0,
                        's_1': 0.02,
                        's_2': 0.03
                        }
    expected_q_mean = 1.2395574782389875
    expected_q_std = 0.13666366485022938
    results_q_mean, results_q_std = get_q_statistics(**input_parameters)
    assert math.isclose(results_q_mean, expected_q_mean, rel_tol=1e-06)
    assert math.isclose(results_q_std, expected_q_std, rel_tol=1e-06)


def test_resolution_function():
    q = 0.2
    mean_q = 0.1
    v_q = 0.01
    expected_result = 2.419707245191433
    result = resolution_function(q, mean_q, v_q)
    assert math.isclose(result, expected_result, rel_tol=1e-06)


def test_reduce_data():
    sans_data = np.ones((10, 10))*10000
    annulus_width = 1.0
    center = (5, 5)
    beamstop_radius = 1.0
    neutron_wavelength = 3.1
    wavelength_spread = 0.1
    detector_resolution = 0.5
    sdd = 480.0
    l1 = 1500.0
    l2 = 100.0
    s1 = 2.54
    s2 = 1.0
    pixel_dim = (0.7, 0.7)
    precision = 9

    reduced_data = reduce_data(sans_data, annulus_width, center, beamstop_radius, neutron_wavelength,
                               wavelength_spread, detector_resolution, sdd, l1, l2, s1, s2, pixel_dim, precision)

    # Assert keys exist in the reduced_data dictionary
    assert 'Q' in reduced_data
    assert 'I' in reduced_data
    assert 'Idev' in reduced_data
    assert 'Qdev' in reduced_data
    assert 'BS' in reduced_data
    assert 'Q_0' in reduced_data
    assert 'reduction_timestamp' in reduced_data

    detector_axis_length = sans_data.shape[0] * pixel_dim[0]
    n_bins = int(detector_axis_length / annulus_width)
    radii = np.linspace(0, detector_axis_length, n_bins)
    skipped_bins = 0
    for r0 in radii:
        if r0 <= annulus_width:
            skipped_bins += 1
            continue
    n_bins -= skipped_bins

    # Assert the shapes of the arrays in the reduced_data dictionary
    assert reduced_data['Q'].shape == (n_bins,)
    assert reduced_data['I'].shape == (n_bins,)
    assert reduced_data['Idev'].shape == (n_bins,)
    assert reduced_data['Qdev'].shape == (n_bins,)
    assert reduced_data['BS'].shape == (n_bins,)
    assert reduced_data['Q_0'].shape == (n_bins,)
    assert bool(datetime.datetime.fromisoformat(reduced_data['reduction_timestamp']))


def test_rescale_with_empty_and_blocked_beams():
    sample_and_cell = np.ones((10, 10))*5
    beam_blocked = np.ones((10, 10))
    empty_cell = np.zeros((10, 10))
    transmission_sample_and_cell = 9.0
    transmission_cell = 9.9
    expected = np.array([[0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455],
                         [0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455,
                          0.54545455, 0.54545455, 0.54545455, 0.54545455, 0.54545455]])
    result = rescale_with_empty_and_blocked_beams(sample_and_cell, beam_blocked,
                                                  empty_cell, transmission_sample_and_cell,
                                                  transmission_cell)

    assert np.allclose(result, expected, rtol=1e-9)
