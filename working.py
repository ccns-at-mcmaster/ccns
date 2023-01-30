from cnbl.loader import *
from cnbl.masking import *
from cnbl.utils import print_impact_matrix
from cnbl.reduction import *

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
    pixel_dim = (data['y_pixel_size'][0], data['x_pixel_size'][0])
    center = (int(data['beam_center_y'][0]), int(data['beam_center_x'][0]))
    sample_transmission = 0.99
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
    solid_angle_correction(data2d, l_2, center, pixel_dim)

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
                                normalize_time=True)

    # Create estimate of incoherent scattering from sample transmission and subtract from the measured values
    incoherent = estimate_incoherent_scattering(l_2, sample_transmission, data2d.shape)
    data2d = np.subtract(data2d, incoherent)

    # Visualize the processed data
    print_impact_matrix(data2d, title="Processed Data")

    from cnbl.reduction import reduce
    reduced_data = reduce(data2d,
                          0.5,
                          center,
                          b_s,
                          wl,
                          wl_spread,
                          sigma_d,
                          l_1,
                          l_2,
                          s_1,
                          s_2,
                          sample_transmission,
                          sample_thickness,
                          pixel_dim)
