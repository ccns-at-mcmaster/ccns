from epics import caget
import numpy as np
from cnbl.reduction import *
from cnbl.loader import *
import matplotlib.pyplot as plt
import math
from cnbl.utils import print_impact_matrix

if __name__ == '__main__':
    filepath = "C:\\Users\\burkeds\\Desktop\\working\\rocktest0000.raw"
    file = get_nexus_file(filepath)
    data = read_sans_raw(file)
    data2d = data['data']
    center = (int(data['beam_center_y'][0]), int(data['beam_center_x'][0]))
    sample_to_detector = data['sdd'][0] * 100

    if not data['metadata_sample_transmission']:
        data['metadata_sample_transmission'] = 0.90

    if not data['metadata_sample_thickness']:
        data['metadata_sample_thickness'] = 0.5

    sample_transmission = data['metadata_sample_transmission']
    sample_thickness = data['metadata_sample_thickness']
    n_bins = 100

    # Assume perfect empty beam
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
    ax.set_xlabel("Q (1/angstrom)")
    ax.set_ylabel("Scattered Intensity (1/cm)")
    ax.set_xscale('log')
    # ax.set_yscale('log')

    plt.show()
