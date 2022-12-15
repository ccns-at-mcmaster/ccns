from epics import caget
import numpy as np
from cnbl.reduction import *
import matplotlib.pyplot as plt
import math
from cnbl.utils import print_impact_matrix

if __name__ == '__main__':
    wavelength = 3.1
    wavelength_spread = 0.17
    detector_offset = 0.0
    source_aperture_diameter = 5.0
    sample_aperture_diameter = 1.27
    source_aperture_to_sample = 537
    sample_to_detector = 800
    sample_transmission = 0.9
    sample_thickness = 0.5

    data1d = caget('sans:mirr2d[sans_det]-Getim2DDAQ_RBV.AVAL')
    data2d = np.reshape(data1d, (147,147), order='F')
    center = (75, 111)
    n_bins = 100

    #Assume perfect empty beam
    empty = np.ones_like(data2d, dtype='float32')
    solid_angle_correction(data2d, center, sample_to_detector)
    pixel_solid_angle = get_pixel_solid_angle(sample_to_detector)
    scale_to_absolute_intensity(data2d, empty, sample_transmission, sample_thickness, pixel_solid_angle)

    intensities, bins = get_intensity_as_a_function_of_radius_in_pixels(data2d, sample_to_detector, center, n_bins)
    radii = [0.7*b for b in bins]
    angles = [math.atan(r/sample_to_detector) for r in radii]
    qs = [4 * np.pi / 3.1 * math.sin(theta / 2) for theta in angles]

    fig, ax = plt.subplots()
    ax.bar(qs[:-1], intensities, align='edge', width=qs[1]-qs[0])
    ax.set_xlabel("Q (ang^-1)")
    ax.set_ylabel("Scattered Intensity (cm^-1)")
    ax.set_xscale('log')

    plt.show()
