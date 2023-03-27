"""
Methods to create interactive figures within Jupyter notebooks.

    author: Devin Burke

(c) Copyright 2023, McMaster University
"""
import matplotlib.pyplot as plt
import threading
import time

class Jupyter_Figure():
    def __init__(self):
        self.fig = plt.figure()
        self.stop = False
        self.threads = []

    def _update(self, attrib, x):
        while True:
            if self.stop is True:
                break
            time.sleep(2)
            if isinstance(attrib, str):
                setattr(self, attrib, x)
            else:
                print("Specify attribute with a string.")

    def create_thread(self, attrib, x):
        self.threads.append(threading.Thread(target=self._update, args=[attrib, x]))

    def stop_updating(self):
        self.stop = True

    def start_updating(self):
        self.stop = False
        for t in self.threads:
            t.start()


if __name__ == "__main__":
    var=10
    box = Jupyter_Figure()

