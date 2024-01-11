"""
Methods to create interactive figures within Jupyter notebooks.

    author: Devin Burke

(c) Copyright 2023, McMaster University
"""
import matplotlib.pyplot as plt
from time import sleep
import datetime


def draw(canvas):
    """
    Static method to use on JupyterFigure objects. This method is blocking and should be called from a thread. It will
    automatically update and redraw the figure every 5 seconds and is stopped by the JupyterFigure.stop_flag attribute.
    """
    while not canvas.stop_flag:
        canvas.update()
        canvas.redraw()
        sleep(5)
    canvas.stop_flag = False


class JupyterFigure:
    """
    Objects of this class create a matplotlib figure to track an objects attribute.
    ...

    Attributes
    ----------
    obj     : object
        Any python object
    attr     : string
        The attribute to track
    stop_flag     : bool
        flag that tells figures.draw to stop.
    y   : list
        A list of attribute values appended to by each call to update()
    x   : list
        The time (in seconds) since the object read the first attribute value in the list.
    figure  : matplotlib.pyplot.Figure
        A matplotlib Figure object which is redrawn by each call to redraw().
    title     : str
        The figure title
    xlabel     : str
        Label on the x-axis
    ylabel     : str
        Label on the y-axis



    Methods
    -------
    clear():
        Clears the figure and x, y data. Sets a new start time.
    update():
        Appends the attribute value to y and the time (in seconds) since the first list element to x.
    redraw():
        Clears and redraws the figure.
    stop():
        Sets the stop_flag attribute which notifies figures.draw to stop.
    """
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        self.x = []
        self.y = []
        self.start_time = None
        self.error = ""
        self.stop_flag = False

        self.title = None
        self.xlabel = None
        self.ylabel = None

        self.figure = plt.figure()
        self.ax = None
        self.line1 = None

        self.update()
        self.redraw()

    def clear(self):
        self.figure.clear()
        self.start_time = datetime.datetime.now()
        self.x.clear()
        self.y.clear()

    def update(self):
        if not self.x:
            self.start_time = datetime.datetime.now()
        try:
            self.y.append(getattr(self.obj, self.attr))
            current_time = datetime.datetime.now()
            diff = (current_time - self.start_time).seconds
            self.x.append(diff)
        except Exception as e:
            self.error = e

    def redraw(self):
        try:
            self.figure.clear()
            self.ax = self.figure.add_subplot()
            self.ax.set_title(self.title)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
            self.line1 = self.ax.plot(self.x, self.y)
        except Exception as e:
            self.error = e

    def stop(self):
        self.stop_flag = True
