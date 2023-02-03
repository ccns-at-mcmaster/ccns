"""
Class to define a generic data writer for saving reduced SANS data.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""

import uuid


class DataWriter:
    """
    A base class to write reduced sans data into an arbitrary format. These formats are defined in their own
    modules found in this package.

    ...

    Attributes
    ----------
    ext      : str
        The file extension.
    filename : str
        The name of the file you intend to write; not including the extension. If not specified, a unique ID will be
        generated.

    Methods
    -------
    set_filename(filename):
        Set the filename attribute of the writer. This method will automatically remove all whitespace from the name and
        replace it with underscores. If there is a decimal in the filename, it will truncate the string from that point.
    """
    def __init__(self, filename=None):
        self.ext = ".out"
        if filename:
            self.filename = ""
            self.set_filename(filename)
        else:
            # noinspection PyTypeChecker
            filename = str(uuid.uuid4())
            self.set_filename(filename)

    def set_filename(self, filename):
        """
        Set the filename attribute of the writer. This method will automatically replace whitespace with underscores.
        If there is a decimal in the filename, it will truncate the string from that point.

        :param filename: A string containing the pre-extension part of the filename.
        :return filename: Returns the filename and extension as a string.
        """
        if not isinstance(filename, str):
            # noinspection PyTypeChecker
            raise Exception("The filename should be a string that does not contain whitespace or decimals.")
        if ' ' in filename:
            filename = filename.replace(" ", "_")
        if '.' in filename:
            filename = filename.split(".")[0]
        self.filename = filename + self.ext
        return filename + self.ext

    def write(self, data):
        pass
