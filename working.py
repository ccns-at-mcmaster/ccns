from cnbl.loader import *
from cnbl.masking import *
from cnbl.utils import print_impact_matrix
from cnbl.reduction import *
from cnbl.writers.nxcansas_writer import get_sasentry, NXcanSASWriter
from cnbl.analysis import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load nexus file and read it
    filepath = "raw_data/agbeh.raw"
    file = get_nexus_file(filepath)
    data = read_sans_raw(file)

    # Instantiate a DataWriter
    path = "C:\\Users\\burkeds\\Desktop\\working\\"
    filename = path + "test"
    writer = NXcanSASWriter(filename)

    # Extract useful information from the data dictionary
    # Create a copy of the 2D impact matrix to manipulate
    data2d = data['data'].copy()
    wl = data['wavelength'][0]
    wl_spread = data['wavelength_spread'][0]
    sigma_d = data['x_resolution'][0]
    b_s = data['beamstop_radius'][0]
    s_1 = data['slit_one'][0]
    s_2 = data['slit_two'][0]
    l_1 = data['collimator_length'][0]
    l_2 = sample_to_detector = data['sample_to_detector'][0] * 100  # Temporary, convert simulated rb to cm
    pixel_dim = (data['y_pixel_size'][0], data['x_pixel_size'][0])
    center = (int(data['beam_center_y'][0]), int(data['beam_center_x'][0]))
    sample_transmission = data['sample_transmission'][0]
    sample_thickness = data['sample_thickness'][0]
    illuminated_sample_area = data['illuminated_sample_area'][0]

    # This is done at the EPICS level, per pixel. I will keep it here now, but it may have to change. Now, efficiency is
    # accounting for at data acquisition and during reduction.
    detector_efficiency = 0.7
    counting_time = 600.0
    monitor_counts = 8.42128E+06
    # Effective beam stop assuming 15 cm between detector and beamstop
    l_b = 15
    b_s = b_s * l_2 / (l_2 - l_b)
    # Visualize the raw data
    # print_impact_matrix(data2d)

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

    # If you have empty and beam blocked matrices, you can rescale your data
    # data2d = rescale_with_empty_and_blocked_beams(sample_and_cell,
    #                                              beam_blocked,
    #                                              empty_cell,
    #                                              transmission_sample_and_cell,
    #                                              transmission_cell)

    # Perform solid angle correction on data
    solid_angle_correction(data2d, l_2, center, pixel_dim)

    # Scale data to absolute intensity
    scale_to_absolute_intensity(data2d,
                                empty,
                                center,
                                l_2,
                                sample_transmission,
                                sample_thickness,
                                sample_to_detector,
                                illuminated_sample_area,
                                detector_efficiency,
                                counting_time,
                                monitor_counts,
                                normalize_time=False,
                                pixel_dim=pixel_dim)

    # Create estimate of incoherent scattering from sample transmission and subtract from the measured values
    incoherent = estimate_incoherent_scattering(l_2, sample_transmission, data2d.shape)
    data2d = np.subtract(data2d, incoherent)

    # Visualize the processed data
    # print_impact_matrix(data2d, title="Processed Data")

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
                          pixel_dim)

    # Generate a Guinier plot and perform a linear regression to calculate 'Rg'
    # methods in cnbl.analysis require an xarray.DataArray object
    x = get_xarray(reduced_data)
    # Ensure Q is monotonically increasing. Because Q is the mean Q of the bin, for small scattering angles it is
    # Q might not increase monotonically. This breaks slicing.
    # Truncating the data using beam-stop shadow factor will also solve this issue.
    x = x.sortby('Q', ascending=True)
    # Plot intensity
    x.sel(name='I').plot()
    plt.show()
    # Retrieve the Porod plot and DataArray. Remember to subtract the incoherent scattering intensity during reduction.
    line, xr = get_standard_plot(data_array=x, name='porod', q_range=slice(0.11, 0.13))
    fit = xr.polyfit(dim='Log(Q)', deg=1)
    slope = float(fit.polyfit_coefficients[0])
    print("The Porod slope is: ", round(slope, 2))

    """
    # Create a nexus file using the DataWriter
    writer.open()
    """
