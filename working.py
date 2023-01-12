from cnbl.reduction import *
from cnbl.loader import *
from cnbl.masking import *
import matplotlib.pyplot as plt
import math
from cnbl.utils import print_impact_matrix

if __name__ == '__main__':
    # Load nexus file and read it
    filepath = "raw_data/agbeh.raw"
    file = get_nexus_file(filepath)
    data = read_sans_raw(file)

    # Extract useful information from the data dictionary
    data2d = data['data']
    center = (int(data['beam_center_y'][0]), int(data['beam_center_x'][0]))
    sample_to_detector = data['sdd'][0] * 100
    pixel_size = data['x_pixel_size'][()][0]
    wavelength = data['monochromator_wavelength'][()][0]
    # counting_time = data['metadata_counting_time'][0]
    counting_time = 600
    # monitor_counts = data['monitor_integral'][0]
    monitor_counts = 8.42128E+06

    # These values may not be in the data dict, they will be implemented in the future
    if 'metadata_sample_transmission' in data:
        if not data['metadata_sample_transmission']:
            data['metadata_sample_transmission'] = 1.0
    else:
        data['metadata_sample_transmission'] = 1.0

    if 'metadata_sample_thickness' in data:
        if not data['metadata_sample_thickness']:
            data['metadata_sample_thickness'] = 0.2
    else:
        data['metadata_sample_thickness'] = 0.2

    if 'metadata_sample_area' in data:
        if not data['metadata_sample_area']:
            data['metadata_sample_area'] = 3.14
    else:
        data['metadata_sample_area'] = 3.14

    if 'detector_efficiency' in data:
        if not data['detector_efficiency']:
            data['detector_efficiency'] = 0.7
    else:
        data['detector_efficiency'] = 0.7

    sample_transmission = data['metadata_sample_transmission']
    sample_thickness = data['metadata_sample_thickness']
    illuminated_sample_area = data['metadata_sample_area']
    efficiency = data['detector_efficiency']

    n_bins = 100

    # Visualize the raw data
    print_impact_matrix(data2d)

    # Demonstrate masking
    rectangular_mask = get_mask('rectangle', x_width=30, y_width=30, origin=(50, 50))
    circular_mask = get_mask('circle', outer_radius=30, origin=(50, 50))
    ring_mask = get_mask('ring', inner_radius=15, outer_radius=30, origin=(50, 50))
    # apply_mask(data2d, ring_mask)
    # print_impact_matrix(data2d)
    
    # Assume empty beam results in one count in each pixel
    empty = np.ones_like(data2d, dtype='float32')

    # Perform solid angle correction on data
    solid_angle_correction(data2d, center, sample_to_detector)

    # Get the solid angle of a pixel as scattering angle zero.
    # Scale data to absolute intensity
    pixel_solid_angle = get_pixel_solid_angle(sample_to_detector)
    scale_to_absolute_intensity(data2d, empty, sample_transmission, sample_thickness, pixel_solid_angle,
                                illuminated_sample_area, efficiency, counting_time, monitor_counts)

    # Create estimate of incoherent scattering from sample transmission and subtract from the measured values
    incoherent = estimate_incoherent_scattering(sample_to_detector, sample_transmission, shape=data2d.shape)
    data2d = np.subtract(data2d, incoherent)

    # Get a histogram of radially averaged intensities
    # Wide angle correction is performed during radial averaging.
    intensities, bins = get_intensity_as_a_function_of_radius_in_pixels(data2d, sample_to_detector, center, n_bins)

    # Convert bins in units of pixels to physical lengths
    radii = [pixel_size*b for b in bins]

    # Convert bins in units of length to scatting angle
    angles = [math.atan(r/sample_to_detector) for r in radii]

    # Convert bins in units of scattering angle to Q value
    qs = [4 * np.pi / wavelength * math.sin(theta / 2) for theta in angles]

    # Create a figure of abs. Intensity vs Q
    fig, ax = plt.subplots()
    ax.bar(qs[:-1], intensities, align='edge', width=qs[1]-qs[0])
    ax.set_xlabel("Q (1/angstrom)")
    ax.set_ylabel("Scattered Intensity (1/cm)")
    ax.set_xscale('log')
    # ax.set_yscale('log')

    plt.show()
