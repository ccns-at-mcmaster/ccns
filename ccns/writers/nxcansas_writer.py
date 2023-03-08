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
import traceback
import uuid


def _h5_float(x):
    """
    Convert a float to a numpy array.

    :param x: The float to be written
    :return: The numpy array
    """
    if isinstance(x, numpy.ndarray):
        return x

    if x is None:
        # noinspection PyTypeChecker
        return numpy.array([0.0], dtype=numpy.float32)

    if not (isinstance(x, list)):
        x = [x]
    # noinspection PyTypeChecker

    return numpy.array(x, dtype=numpy.float32)


def _h5_string(string):
    """
    Convert a string to a numpy string in a numpy array. This way it is written to the HDF5 file as a fixed length
    ASCII string.

    :param string: The string to be converterd to an array.
    :return: A numpy array of the string.
    """
    if string is None:
        string = "none"

    if isinstance(string, numpy.ndarray):
        return string

    elif not isinstance(string, str):
        string = str(string)

    return numpy.array([numpy.string_(string)])


def get_empty_sasentry():
    """
    Returns an 'empty' python dictionary with a structure similar to NXcanSAS. Each key in this dict is expected by
    methods in NXcanSASWriter to write an NXcanSAS compliant HDF5 nexus entry to a nexus file.

    :return: The 'empty' NXcanSAS dictionary.
    """
    # noinspection PyTypeChecker
    sas_entry = {'entry_name': str(uuid.uuid4()),
                 'title': 'none',
                 'run': 0,
                 'run_name': 'none',
                 'reduction_timestamp': 'none',
                 'mask': numpy.empty(0),
                 'I': numpy.empty(0),
                 'Idev': numpy.empty(0),
                 'Q': numpy.empty(0),
                 'Qdev': numpy.empty(0),
                 'ShadowFactor': numpy.empty(0),
                 'slit_one': 0,
                 'slit_two': 1,
                 'collimator_length': 0,
                 'collimator_to_sample': 0,
                 'instrument_name': 'none',
                 'sample_to_detector': 0,
                 'detector_height': 0,
                 'beam_center_x': 0,
                 'beam_center_y': 0,
                 'x_pixel_size': 0,
                 'y_pixel_size': 0,
                 'source_probe': 'none',
                 'source_type': 'none',
                 'wavelength': 0,
                 'wavelength_spread': 0,
                 'sample_name': 'none',
                 'sample_thickness': 0,
                 'sample_transmission': 0,
                 'sample_temperature': 0,
                 'sample_details': 0,
                 'sample_x_position': 0,
                 'sample_y_position': 0,
                 'process_name': 'none',
                 'process_date': 'none',
                 'process_description': 'none',
                 'process_term': 'none',
                 'process_note': 'none',
                 'process_collection': {},
                 'collection': {}}
    return sas_entry


def get_sasentry(raw_data, reduced_data, entry=None):
    """

    :param raw_data: A dictionary of raw sans_data. This is usually returned by ccns.loader.read_sans_raw.
    :param reduced_data: A dictionary of reduced sans data. This is usually the dictionary returned by
                         ccns.reduction.reduce.
    :param entry: If not specified, this method fetches an 'empty' NXcanSAS entry dictionary from get_empty_sasentry. If
                  a dictionary is passed as entry, the dictionary is updated with key, value pairs from raw_data and
                  reduced_data.
    :return: The updated entry dictionary ready to be passed to NXcanSASWriter.add_entry.
    """
    if entry is None:
        entry = get_empty_sasentry()
    for key in entry.keys():
        entry[key] = raw_data.get(key, entry[key])
        entry[key] = reduced_data.get(key, entry[key])
    return entry


class NXcanSASWriter(DataWriter):
    """
    An inheritor class to write reduced sans data into NXcanSAS, an HDF5 nexus format. An instance of this class can
    create or open a single nexus file and add, remove, or edit any number of entries to the file. Any entries added to
    the file must be NXcanSAS format (https://manual.nexusformat.org/classes/applications/NXcanSAS.html). The easiest
    way to add an entry is to pass a python dictionary similar to that found in get_empty_sasentry within this module.
    This 'empty' dict can be 'filled' by passing raw and reduced data dicts to get_sasentry. The returned dict can then
    be passed to NXcanSASWriter.add_sasentry and the writer will automatically format and write an NXcanSAS entry to
    the file.

    ...

    Attributes
    ----------
    entries      : dict
        A dictionary of nexus entries in the currently open file where keys are the entry names and values are h5py
        groups (entries).
    ext      : str
        The file extension.
    filename : str
        The name of the file you intend to write; not including the extension. If not specified, a unique ID will be
        generated.
    nexus      : h5py._hl.files.File
        An HDF5 nexus file opened with h5py.
    file_open      : bool
        A boolean that's True if a nexus file is open and false otherwise.

    Methods
    -------
    set_filename(filename):
        Set the filename attribute of the writer. This method will automatically remove all whitespace from the name and
        replace it with underscores. If there is a decimal in the filename, it will truncate the string from that point.
    close():
        Closes the currently open nexus file.
    open(fname):
        Opens a nexus file at location 'fname' if it exists or creates a nexus file if 'fname' is not specified. The
        name of the newly created nexus file is taken from the 'filename' attribute.
    add_entry(entry):
        If passed a nexus object, this object is written to the file. If a NXcanSAS valid dictionary is passed as
        'entry', a nexus object is created and written to file.
    delete_entry(entry):
        If a nexus object is passed as 'entry', that object is deleted from the file if present. If a string is passed
        as 'entry', the entry in the file with the name that matches the string is deleted.
    """

    def __init__(self, filename=None):
        DataWriter.__init__(self, filename)
        self.entries = {}
        self.ext = ".nxs"
        self.set_filename(filename)
        self.nexus = None
        self.file_open = False
        return

    def close(self):
        if self.file_open is False:
            print("A nexus file is not currently open.")
            return
        if self.nexus is None:
            # noinspection PyTypeChecker
            print("You need to first open a nexus file before it can be closed.")
            return
        self.nexus.close()
        self.file_open = False
        return

    def open(self, fname=None):
        if self.file_open:
            print("A nexus file is already open.")
            return
        if fname is None:
            self.nexus = h5py.File(self.filename, "w-")
            self.nexus.attrs["filename"] = self.filename
            self.nexus.attrs["file_time"] = datetime.datetime.now().isoformat()
            self.nexus.attrs["creator"] = "NXcanSASWriter.open()"
            self.nexus.attrs["H5PY_VERSION"] = h5py.__version__
            self.entries.clear()
            self.file_open = True
        else:
            self.nexus = h5py.File(fname, "r+")
            self.entries.clear()
            for entry_name in self.nexus.keys():
                self.entries[entry_name] = self.nexus[entry_name]
            self.file_open = True
        return

    def _write_entry(self, entry):
        name = entry['entry_name']
        try:
            # /entry
            sasentry = self.nexus.create_group(name)
            sasentry.attrs["canSAS_class"] = "SASentry"
            sasentry.attrs["version"] = "1.1"
            sasentry.attrs["default"] = "data"
            self.nexus['/{}/definition'.format(name)] = _h5_string("NXcanSAS")
            self.nexus['/{}/title'.format(name)] = _h5_string(entry.get('title', 'none'))
            self.nexus['/{}/run'.format(name)] = entry.get('run', 0)
            self.nexus['/{}/run'.format(name)].attrs['name'] = _h5_string(entry.get('run_name', 'none'))

            # /entry/data
            sasdata = sasentry.create_group("data")
            sasdata.attrs["canSAS_class"] = "SASdata"
            sasdata.attrs["signal"] = "I"
            sasdata.attrs["I_axes"] = ["Q"]
            sasdata.attrs["Q_indices"] = [0]
            # Expects array where false is no mask and true is mask. I need to change
            # this in the masking methods.
            sasdata.attrs["mask"] = "mask"
            sasdata.attrs["Mask_indices"] = [0]
            # noinspection PyArgumentList
            sasdata.attrs["reduction_timestamp"] = _h5_string(entry['reduction_timestamp'])

            self.nexus['/{}/data/mask'.format(name)] = entry.get("mask", numpy.zeros_like(entry['I']))

            # Need to rename Q_0 to Q in reduced data dict
            self.nexus['/{}/data/Q'.format(name)] = entry["Q"]
            self.nexus['/{}/data/Q'.format(name)].attrs["units"] = "1/angstrom"
            self.nexus['/{}/data/Q'.format(name)].attrs["resolutions"] = "Qdev"
            self.nexus['/{}/data/Q'.format(name)].attrs["resolutions_description"] = "Gaussian"

            self.nexus['/{}/data/I'.format(name)] = entry["I"]
            self.nexus['/{}/data/I'.format(name)].attrs["units"] = "1/cm"
            self.nexus['/{}/data/I'.format(name)].attrs["uncertainties"] = "Idev"
            self.nexus['/{}/data/I'.format(name)].attrs["scaling_factor"] = "ShadowFactor"

            self.nexus['/{}/data/Idev'.format(name)] = entry["Idev"]
            self.nexus['/{}/data/Idev'.format(name)].attrs["units"] = "1/cm"

            self.nexus['/{}/data/Qdev'.format(name)] = entry["Qdev"]
            self.nexus['/{}/data/Qdev'.format(name)].attrs["units"] = "1/angstrom"

            self.nexus['/{}/data/ShadowFactor'.format(name)] = entry["BS"]

            # /entry/instrument
            sasinstrument = sasentry.create_group("instrument")
            sasinstrument.attrs["canSAS_class"] = "SASinstrument"

            pinhole1 = sasinstrument.create_group("slit_one")
            pinhole1.attrs["NX_class"] = "NXpinhole"
            # nexus['/{}/instrument/slit_one/depends_on'] = '.'
            self.nexus['/{}/instrument/slit_one/diameter'.format(name)] = _h5_float(entry['slit_one'])
            pinhole2 = sasinstrument.create_group("slit_two")
            pinhole2.attrs["NX_class"] = "NXpinhole"
            self.nexus['/{}/instrument/slit_two/diameter'.format(name)] = _h5_float(entry['slit_two'])
            # nexus['/{}/instrument/slit_two/depends_on'] = './collimator'

            collimator = sasinstrument.create_group("collimator")
            collimator.attrs["canSAS_class"] = "SAScollimation"
            self.nexus['/{}/instrument/collimator/length'.format(name)] = _h5_float(entry['collimator_length'])
            self.nexus['/{}/instrument/collimator/distance'.format(name)] = _h5_float(entry['collimator_to_sample'])

            detector = sasinstrument.create_group("detector")
            detector.attrs["canSAS_class"] = "SASdetector"
            self.nexus['/{}/instrument/detector/name'.format(name)] = _h5_string(entry['instrument_name'])
            self.nexus['/{}/instrument/detector/SDD'.format(name)] = _h5_float(entry['sample_to_detector'])
            self.nexus['/{}/instrument/detector/x_position'.format(name)] = _h5_float(0.0)
            self.nexus['/{}/instrument/detector/y_position'.format(name)] = _h5_float(entry['detector_height'])
            self.nexus['/{}/instrument/detector/beam_center_x'.format(name)] = _h5_float(entry['beam_center_x'])
            self.nexus['/{}/instrument/detector/beam_center_x'.format(name)].attrs["units"] = "pixels"
            self.nexus['/{}/instrument/detector/beam_center_y'.format(name)] = _h5_float(entry['beam_center_y'])
            self.nexus['/{}/instrument/detector/beam_center_y'.format(name)].attrs["units"] = "pixels"
            self.nexus['/{}/instrument/detector/x_pixel_size'.format(name)] = _h5_float(entry['x_pixel_size'])
            self.nexus['/{}/instrument/detector/x_pixel_size'.format(name)].attrs["units"] = "cm"
            self.nexus['/{}/instrument/detector/y_pixel_size'.format(name)] = _h5_float(entry['y_pixel_size'])
            self.nexus['/{}/instrument/detector/y_pixel_size'.format(name)].attrs["units"] = "cm"

            source = sasinstrument.create_group("source")
            source.attrs["canSAS_class"] = "SASsource"
            self.nexus['/{}/instrument/source/probe'.format(name)] = _h5_string(entry['source_probe'])
            self.nexus['/{}/instrument/source/type'.format(name)] = _h5_string(entry['source_type'])
            self.nexus['/{}/instrument/source/incident_wavelength'.format(name)] = _h5_string(entry['wavelength'])
            self.nexus['/{}/instrument/source/incident_wavelength'.format(name)].attrs["units"] = "angstrom"
            self.nexus['/{}/instrument/source/incident_wavelength_spread'.format(name)] = \
                _h5_string(entry['wavelength_spread'])

            sample = sasentry.create_group("sample")
            sample.attrs["canSAS_class"] = "SASsample"
            self.nexus['/{}/sample/name'.format(name)] = _h5_string(entry['sample_name'])
            self.nexus['/{}/sample/thickness'.format(name)] = _h5_string(entry['sample_thickness'])
            self.nexus['/{}/sample/thickness'.format(name)].attrs["units"] = "cm"
            self.nexus['/{}/sample/transmission'.format(name)] = _h5_string(entry['sample_transmission'])
            self.nexus['/{}/sample/temperature'.format(name)] = _h5_string(entry['sample_temperature'])
            self.nexus['/{}/sample/temperature'.format(name)].attrs["units"] = "K"
            self.nexus['/{}/sample/details'.format(name)] = _h5_string(entry['sample_details'])
            self.nexus['/{}/sample/x_position'.format(name)] = _h5_float(entry['sample_x_position'])
            self.nexus['/{}/sample/x_position'.format(name)].attrs["units"] = "cm"
            self.nexus['/{}/sample/y_position'.format(name)] = _h5_float(entry['sample_y_position'])
            self.nexus['/{}/sample/y_position'.format(name)].attrs["units"] = "cm"

            process = sasentry.create_group("process")
            process.attrs["canSAS_class"] = "SASprocess"
            self.nexus['/{}/process/name'.format(name)] = _h5_string(entry['process_name'])
            self.nexus['/{}/process/date'.format(name)] = _h5_string(entry['process_date'])
            self.nexus['/{}/process/description'.format(name)] = _h5_string(entry['process_description'])
            if isinstance(entry['process_term'.format(name)], str):
                self.nexus['/{}/process/term'.format(name)] = _h5_string(entry['process_term'])
            if isinstance(entry['process_term'.format(name)], float) or isinstance(entry['process_term'], int):
                self.nexus['/{}/process/term'.format(name)] = _h5_float(entry['process_term'])
            self.nexus['/{}/process/note'.format(name)] = _h5_string(entry['process_note'])
            process_collection = process.create_group("collection")
            process_collection.attrs["canSAS_class"] = "SASprocessnote"
            if entry['process_collection'] is not None:
                for k, v in entry['process_collection']:
                    try:
                        process_collection.create_dataset(k, data=v)
                    except Exception:
                        process_collection.create_dataset(k, data=str(v))

            collection = sasentry.create_group("COLLECTION")
            collection.attrs["canSAS_class"] = "SASnote"

            if entry['collection'] is not None:
                for k, v in entry['collection']:
                    try:
                        process_collection.create_dataset(k, data=v)
                    except Exception:
                        process_collection.create_dataset(k, data=str(v))

        except Exception as e:
            # noinspection PyTypeChecker
            print(traceback.format_exc())
            # noinspection PyTypeChecker
            print("Failed to write nexus file: " + str(e))
            if self.nexus[name]:
                del self.nexus[name]
            return
        return sasentry

    def add_entry(self, entry=None):
        if (self.nexus is None) or (self.file_open is False):
            print("You must first open a file with open() before you can add an entry")
            return
        if isinstance(entry, h5py._hl.group.Group):
            print("Appending to entries")
            self.entries[entry.name] = entry
            return
        if entry is None:
            entry = get_empty_sasentry()
        if entry['entry_name'] in self.entries:
            # noinspection PyTypeChecker
            print('An entry called {} is already in the list of entries.'.format(entry['entry_name']))
            return
        if isinstance(entry, dict):
            sas_entry = self._write_entry(entry)
            self.entries[sas_entry.name] = sas_entry
        return

    def delete_entry(self, arg):
        if self.file_open is False:
            print("A nexus file must be open to delete entries.")
            return
        if isinstance(arg, str):
            try:
                del self.nexus[arg]
                del self.entries[arg]
            except KeyError:
                print("There is no entry with name: {}".format(arg))
            finally:
                return
        if isinstance(arg, h5py._hl.group.Group):
            try:
                del self.nexus[arg.name]
                del self.entries[arg.name]
            except KeyError:
                print("{} is not an entry in the currently open nexus file.".format(arg.name))
                return
            finally:
                return
