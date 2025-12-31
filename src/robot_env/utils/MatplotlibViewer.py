import numpy as np
import matplotlib.pyplot as plt


class MatplotlibViewer:
    def __init__(self, config):
        self.config = config
        self.fig, self.ax = None, None
        self.image = None
        self.fps   = 60

    def initialize(self):
        self.fig, self.ax = plt.subplots(
            nrows   = 1,
            ncols   = 1,
            figsize = (self.config.viewer.width, self.config.viewer.height),
        )

    def start_interactive(self):
        plt.ion()

    def stop_interactive(self):
        plt.ioff()

    def show(self, rgb):
        if self.image is None:
            self.image = self.ax.imshow(rgb)
        else:
            self.image.set_data(rgb)
        plt.draw()
        plt.pause(1 / self.fps)

