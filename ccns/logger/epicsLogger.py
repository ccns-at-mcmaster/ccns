"""
Simple class for an EPICS logger.

    Author: Devin Burke

(c) Copyright 2023, McMaster University
"""
import os
import datetime
import csv
from epics import PV
from time import sleep
from threading import Thread


class EpicsLogger:
    """
    Creates an EPICS record field logger. This class defines a logger object which can take any number of EPICS process
    variables and write the values of these fields to a .csv file at regular intervals.

    ...

    Attributes
    ----------
    filetime      : int
        The interval of time in seconds that the .csv log file spans. A new file is written every 'filetime' seconds.
    interval      : int
        The interval of time between calls to read the PVs.
    status        : string
        Is 'idle' when not logging and is 'writing' when logging PVs.
    stop_flag     : bool
        If True, logging is stopped.
    pvs           : dict
        A dictionary where keys are PV names and values are pyepics PV objects.

    Methods
    -------
    start():
        Starts the logger.
    stop():
        Stops the logger by writing 1 to stop_flag.
    """
    def __init__(self, pv_names=None):
        self.filetime = 3600
        self.interval = 5
        self.status = 'idle'
        self.stop_flag = 0
        self.pvs = {}
        if not pv_names:
            raise Exception("A list of PV names must be specified.")
        for pv in pv_names:
            self.pvs[pv] = PV(pv)

    def start(self):
        Thread(target=self._logger).start()

    def stop(self):
        self.stop_flag = 1

    def _logger(self):
        self.status = 'writing'
        while True:
            start = datetime.datetime.now()
            filename = str(start.day) + str(start.month) + str(start.year) + '_' \
                + str(start.hour) + '-' + str(start.minute) + '-' + str(start.second)
            filename = os.getcwd() + '\\epicslog_' + filename + '.csv'
            self._newfile(start, filename)
            if self.stop_flag:
                self.stop_flag = 0
                break
        self.status = 'idle'

    def _newfile(self, start, filename):
        header = list(self.pvs.keys())
        header.insert(0, str(start.isoformat()))
        self._writer(filename, header)
        while True:
            now = datetime.datetime.now()
            delta_s = now - start
            delta_s = delta_s.seconds
            if (delta_s >= self.filetime) or self.stop_flag:
                break
            if delta_s % self.interval == 0:
                values = [pv.value for pv in self.pvs.values()]
                timestamp = now.isoformat()
                values.insert(0, timestamp)
                Thread(target=self._writer, args=[filename, values]).start()
                sleep(1)

    @staticmethod
    def _writer(filename, row):
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(row)
