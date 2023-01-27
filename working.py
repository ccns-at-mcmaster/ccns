from cnbl.reduction.scattering import *
from cnbl.loader import *
from cnbl.masking import *
import matplotlib.pyplot as plt
import math
from cnbl.utils import print_impact_matrix
import numpy as np
from cnbl.reduction.gaussiandq import get_q_statistics

if __name__ == '__main__':
    # Load nexus file and read it
    filepath = "raw_data/agbeh.raw"
    file = get_nexus_file(filepath)
    data = read_sans_raw(file)

    # Extract useful information from the data dictionary
    data2d = data['data'].copy()
    # Get useful metadata for reduction
    wl = 5.0
    wl_spread = 0.14
    sigma_d = 0.9
    b_s = beamstop_radius = 2.5
    s_1 = slit_one = 5.08
    s_2 = slit_two = 1.91
    l_1 = source_to_sample = 1600
    l_2 = sample_to_detector = 508
    pixel_size = data['x_pixel_size'][0]
    center = (int(data['beam_center_y'][0]), int(data['beam_center_x'][0]))
    sample_transmission = 1.0
    sample_thickness = 0.2
    illuminated_sample_area = 3.14

    # This is done at the EPICS level, per pixel. I will keep it here now, but it may have to change. Now, efficiency is
    # accounting for at data acquisition and during reduction.
    detector_efficiency = 0.7

    counting_time = 600.0
    monitor_counts = 8.42128E+06

    # Visualize the raw data
    print_impact_matrix(data2d)

    # Demonstrate masking
    rectangular_mask = get_mask('rectangle', x_width=30, y_width=30, origin=(50, 50))
    circular_mask = get_mask('circle', outer_radius=30, origin=(50, 50))
    ring_mask = get_mask('ring', inner_radius=15, outer_radius=30, origin=(50, 50))
    masked_pixel_list = [(50, 51), (51, 50), (52, 50), (52, 51), (52, 52), (53, 50), (53, 51), (53, 52)]
    irregular_mask = get_mask('irregular', (147, 147), irregular_pixels=masked_pixel_list)
    # apply_mask(data2d, ring_mask)
    # print_impact_matrix(data2d)

    # Trim 5 pixels from the edges of the data
    trim_edges(data2d, 5)
    
    # Assume empty beam results in one count in each pixel
    empty = np.ones_like(data2d, dtype='float32')

    # Perform solid angle correction on data
    solid_angle_correction(data2d, l_2, center, pixel_size)

    # Scale data to absolute intensity
    scale_to_absolute_intensity(data2d,
                                empty,
                                sample_transmission,
                                sample_thickness,
                                sample_to_detector,
                                illuminated_sample_area,
                                detector_efficiency,
                                counting_time,
                                monitor_counts,
                                normalize_time=False)

    # Create estimate of incoherent scattering from sample transmission and subtract from the measured values
    incoherent = estimate_incoherent_scattering(l_2, sample_transmission, data2d.shape)
    data2d = np.subtract(data2d, incoherent)

    # Visualize the processed data
    print_impact_matrix(data2d, title="Processed Data")

    # Start analysis
    d_r = annulus_width = 0.5
    # Generate list of annular radii
    detector_axis_length = data2d.shape[0] * pixel_size
    n_bins = int(detector_axis_length / annulus_width)
    radii = numpy.linspace(0, detector_axis_length, n_bins)

    reduced_data = {'Q': numpy.empty(1),
                    'scattered_intensity': numpy.empty(1),
                    'scattered_intensity_std': numpy.empty(1),
                    '<Q>': numpy.empty(1),
                    'Q_variance': numpy.empty(1),
                    'BS': numpy.empty(1, dtype=int)}
    ordinate = numpy.empty(1)
    for r_0 in radii:
        if r_0 == 0.0:
            continue
        q, v_q = get_q_statistics(r_0, d_r, b_s, wl, wl_spread, sigma_d, l_1, l_2, s_1, s_2)
        reduced_data['Q'] = numpy.append(reduced_data['Q'], q)
        reduced_data['Q_variance'] = numpy.append(reduced_data['Q_variance'], v_q)

        intensity, intensity_std = get_scattered_intensity(data2d, center, r_0, d_r)
        reduced_data['scattered_intensity'] = numpy.append(reduced_data['scattered_intensity'], intensity)
        reduced_data['scattered_intensity_std'] = numpy.append(reduced_data['scattered_intensity_std'], intensity_std)

        bs_factor = get_beam_stop_factor(r_0, d_r, b_s)
        reduced_data['BS'] = numpy.append(reduced_data['BS'], bs_factor)

        ordinate = numpy.append(ordinate, r_0)
