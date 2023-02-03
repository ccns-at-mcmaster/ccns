"""
Methods to write reduced data to a nexus HDF5 format.
This format is a derivative of the standard NXcanSAS format.
https://manual.nexusformat.org/classes/applications/NXcanSAS.html

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

from .dataWriter import DataWriter
import h5py
import datetime
import numpy

nxcansas_dict = {
    'title': '/entry/title',
    'start_time': '/entry/start_time',
    'end_time': '/entry/end_time',
    'instrument_name': '/entry/instrument/name',
    'source_type': '/entry/instrument/source/source_type',
    'source_name': '/entry/instrument/source/source_name',
    'source_probe': '/entry/instrument/source/source_probe',
    'wavelength': '/entry/instrument/monochromator/wavelength',
    'wavelength_spread': '/entry/instrument/monochromator/wavelength_spread',
    'collimator_length': '/entry/instrument/collimator/geometry/shape/size',
    'slit_one': '/entry/instrument/collimator/slit_one',
    'slit_two': '/entry/instrument/collimator/slit_two',
    'sample_to_detector': '/entry/instrument/detector/distance',
    'detector_height': '/entry/instrument/detector/height',
    'x_pixel_size': '/entry/instrument/detector/x_pixel_size',
    'y_pixel_size': '/entry/instrument/detector/y_pixel_size',
    'beam_center_x': '/entry/instrument/detector/beam_center_x',
    'beam_center_y': '/entry/instrument/detector/beam_center_y',
    'x_resolution': '/entry/instrument/detector/x_resolution',
    'y_resolution': '/entry/instrument/detector/y_resolution',
    'beamstop_radius': '/entry/instrument/detector/beamstop_radius',
    'sample_name': '/entry/sample/sample_name',
    'sample_transmission': '/entry/sample/transmission',
    'sample_thickness': '/entry/sample/thickness',
    'illuminated_sample_area': '/entry/sample/illuminated_sample_area',
    'sample_temperature': '/entry/sample/temperature',
    'monitor_mode': '/entry/monitor/mode',
    'monitor_preset': '/entry/monitor/preset',
    'monitor_integral': '/entry/monitor/integral',
    'data': '/entry/data/data',
    'counting_time': '/entry/instrument/metadata/counting_time'
}


def _h5_float(x):
    """
    Convert a float to a numpy array.

    :param x: The float to be written
    :return: The numpy array
    """
    if not (isinstance(x, list)):
        x = [x]
    # noinspection PyTypeChecker
    return numpy.array(x, dtype=numpy.float32)


def _h5_bool(x):
    """
    Convert a bool to a numpy array.

    :param x: The float to be written
    :return: The numpy array
    """
    if not (isinstance(x, list)):
        x = [x]
    # noinspection PyTypeChecker
    return numpy.array(x, dtype=numpy.bool_)


def _h5_string(string):
    """
    Convert a string to a numpy string in a numpy array. This way it is written to the HDF5 file as a fixed length
    ASCII string.

    :param string: The string to be converterd to an array.
    :return: A numpy array of the string.
    """
    if isinstance(string, numpy.ndarray):
        return string
    elif not isinstance(string, str):
        string = str(string)
    return numpy.array([numpy.string_(string)])


class NXcanSASWriter(DataWriter):
    """
    A DataWriter inheritor class that writes HDF5 nexus files in the NXcanSAS format.
    https://manual.nexusformat.org/classes/applications/NXcanSAS.html

    """

    def __init__(self, filename=None):
        DataWriter.__init__(self, filename)
        self.ext = ".nxs"
        self.set_filename(filename)

    # def set_filename(self, filename):
    #    super().set_filename(filename)

    def write(self, reduced_data):
        nexus = h5py.File(self.filename, "w")

        # /
        nexus.attrs["filename"] = self.filename
        nexus.attrs["file_time"] = datetime.datetime.now().isoformat()
        nexus.attrs["creator"] = "NXcanSASWriter.write()"
        nexus.attrs["H5PY_VERSION"] = h5py.__version__

        # /entry
        sasentry = nexus.create_group("entry")
        sasentry.attrs["canSAS_class"] = "SASentry"
        sasentry.attrs["version"] = "1.1"
        sasentry.attrs["default"] = "sasdata"
        nexus['/entry/definition'] = _h5_string("NXcanSAS")
        # Need to add to reduced data
        nexus['/entry/title'] = _h5_string(reduced_data.get('title', 'none'))
        # Need to add to reduced data
        nexus['/entry/run'] = _h5_string(reduced_data.get('run', 'none'))
        if reduced_data['run_name']:
            nexus['/entry/run'].attrs['name'] = _h5_string(reduced_data['run_name'])
        else:
            nexus['/entry/run'].attrs['name'] = _h5_string('none')

        # /entry/data
        sasdata = nexus.create_group("data")
        sasdata.attrs["canSAS_class"] = "SASdata"
        sasdata.attrs["signal"] = "I"
        sasdata.attrs["I_axes"] = ["Temperature", "Q", "Q", "Q", "Q"]
        sasdata.attrs["Q_indices"] = [1, 2, 3, 4, 5]
        # Expects array where false is no mask and true is mask. I need to change
        # this in the masking methods.
        sasdata.attrs["mask"] = "data_mask"
        sasdata.attrs["Mask_indices"] = [1, 2, 3, 4]
        # Should add key for reduction_timestamp
        # noinspection PyArgumentList
        sasdata.attrs["reduction_timestamp"] = _h5_string(reduced_data['reduction_timestamp'])

        # Need to rename Q_0 to Q in reduced data dict
        nexus['/entry/data/Q'] = reduced_data["Q"]
        nexus['/entry/data/Q'].attrs["units"] = "1/angstrom"
        nexus['/entry/data/Q'].attrs["resolutions"] = "Qdev"
        nexus['/entry/data/Q'].attrs["resolutions_description"] = "Gaussian"

        nexus['/entry/data/I'] = reduced_data["scattered_intensity"]
        nexus['/entry/data/I'].attrs["units"] = "1/cm"
        nexus['/entry/data/I'].attrs["uncertainties"] = "Idev"
        nexus['/entry/data/I'].attrs["scaling_factor"] = "ShadowFactor"

        nexus['/entry/data/Idev'] = reduced_data["scattered_intensity_std"]
        nexus['/entry/data/Idev'].attrs["units"] = "1/cm"

        # Need to change name of Q_variance key in reduced data
        nexus['/entry/data/Qdev'] = reduced_data["Qdev"]
        nexus['/entry/data/Qdev'].attrs["units"] = "1/angstrom"

        nexus['/entry/data/ShadowFactor'] = _h5_bool(reduced_data["BS"])

        # /entry/instrument
        sasinstrument = sasentry.create_group("instrument")
        sasinstrument.attrs["canSAS_class"] = "SASinstrument"

        pinhole1 = sasinstrument.create_group("slit_one")
        pinhole1.attrs["NX_class"] = "NXpinhole"
        # nexus['/entry/instrument/slit_one/depends_on'] = '.'
        # Need to add slit metadata to reduced data dict
        nexus['/entry/instrument/slit_one/diameter'] = _h5_float(reduced_data['slit_one'])
        pinhole2 = sasinstrument.create_group("slit_two")
        pinhole2.attrs["NX_class"] = "NXpinhole"
        nexus['/entry/instrument/slit_two/diameter'] = _h5_float(reduced_data['slit_two'])
        # nexus['/entry/instrument/slit_two/depends_on'] = './collimator'

        collimator = sasinstrument.create_group("collimator")
        collimator.attrs["canSAS_class"] = "SAScollimation"
        nexus['/entry/instrument/collimator/length'] = _h5_float(reduced_data['collimator_length'])
        # Need to add collimator to sample distance
        nexus['/entry/instrument/collimator/distance'] = _h5_float(reduced_data['collimator_to_sample'])

        detector = sasinstrument.create_group("detector")
        detector.attrs["canSAS_class"] = "SASdetector"
        nexus['/entry/instrument/detector/name'] = _h5_string(reduced_data['instrument_name'])
        nexus['/entry/instrument/detector/SDD'] = _h5_float(reduced_data['sample_to_detector'])
        nexus['/entry/instrument/detector/x_position'] = _h5_float(0.0)
        nexus['/entry/instrument/detector/y_position'] = _h5_float(reduced_data['detector_height'])
        nexus['/entry/instrument/detector/beam_center_x'] = _h5_float(reduced_data['beam_center_x'])
        nexus['/entry/instrument/detector/beam_center_x'].attrs["units"] = "pixels"
        nexus['/entry/instrument/detector/beam_center_y'] = _h5_float(reduced_data['beam_center_y'])
        nexus['/entry/instrument/detector/beam_center_y'].attrs["units"] = "pixels"
        nexus['/entry/instrument/detector/x_pixel_size'] = _h5_float(reduced_data['x_pixel_size'])
        nexus['/entry/instrument/detector/x_pixel_size'].attrs["units"] = "cm"
        nexus['/entry/instrument/detector/y_pixel_size'] = _h5_float(reduced_data['y_pixel_size'])
        nexus['/entry/instrument/detector/y_pixel_size'].attrs["units"] = "cm"

        source = sasinstrument.create_group("source")
        source.attrs["canSAS_class"] = "SASsource"
        nexus['/entry/instrument/source/probe'] = _h5_string(reduced_data['source_probe'])
        nexus['/entry/instrument/source/type'] = _h5_string(reduced_data['source_type'])
        nexus['/entry/instrument/source/incident_wavelength'] = _h5_string(reduced_data['wavelength'])
        nexus['/entry/instrument/source/incident_wavelength'].attrs["units"] = "angstrom"
        nexus['/entry/instrument/source/incident_wavelength_spread'] = _h5_string(reduced_data['wavelength_spread'])

        sample = sasentry.create_group("sample")
        sample.attrs["canSAS_class"] = "SASsample"
        nexus['/entry/sample/name'] = _h5_string(reduced_data['sample_name'])
        nexus['/entry/sample/thickness'] = _h5_string(reduced_data['sample_thickness'])
        nexus['/entry/sample/thickness'].attrs["units"] = "cm"
        nexus['/entry/sample/transmission'] = _h5_string(reduced_data['sample_transmission'])
        nexus['/entry/sample/temperature'] = _h5_string(reduced_data['sample_temperature'])
        nexus['/entry/sample/temperature'].attrs["units"] = "K"
        # Need to add sample_details key. I think I call this description atm.
        nexus['/entry/sample/details'] = _h5_string(reduced_data['sample_details'])
        # Need to add sample position keys
        nexus['/entry/sample/x_position'] = _h5_float(reduced_data['sample_x_position'])
        nexus['/entry/sample/x_position'].attrs["units"] = "cm"
        nexus['/entry/sample/y_position'] = _h5_float(reduced_data['sample_y_position'])
        nexus['/entry/sample/y_position'].attrs["units"] = "cm"

        # Need to add keys for all these process variables
        process = sasentry.create_group("process")
        process.attrs["canSAS_class"] = "SASprocess"
        nexus['/entry/process/name'] = _h5_string(reduced_data['process_name'])
        nexus['/entry/process/date'] = _h5_string(reduced_data['process_date'])
        nexus['/entry/process/description'] = _h5_string(reduced_data['process_description'])
        if isinstance(reduced_data['process_term'], str):
            nexus['/entry/process/term'] = _h5_string(reduced_data['process_term'])
        if isinstance(reduced_data['process_term'], float) or isinstance(reduced_data['process_term'], int):
            nexus['/entry/process/term'] = _h5_float(reduced_data['process_term'])
        nexus['/entry/process/note'] = _h5_string(reduced_data['process_note'])
        process_collection = process.create_group("collection")
        process_collection.attrs["canSAS_class"] = "SASprocessnote"
        for k, v in reduced_data['process_collection']:
            try:
                process_collection.create_dataset(k, data=v)
            except Exception:
                process_collection.create_dataset(k, data=str(v))

        collection = sasentry.create_group("COLLECTION")
        collection.attrs["canSAS_class"] = "SASnote"
        for k, v in reduced_data['collection']:
            try:
                process_collection.create_dataset(k, data=v)
            except Exception:
                process_collection.create_dataset(k, data=str(v))

        nexus.close()
        return
