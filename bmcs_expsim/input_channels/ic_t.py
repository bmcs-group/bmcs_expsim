import os
import numpy as np
from traits.api import HasTraits, Float, Array, Str, Directory

class IC(HasTraits):
    name = Str
    delta_T = Float(0.0)
    offset = Array(Float, value=[0.0, 0.0, 0.0])
    rotation = Array(Float, value=[0.0, 0.0, 0.0])
    master_dir = Directory

    def synchronize_time(self, universal_time):
        return universal_time + self.delta_T

class PointDataIC(IC):
    channel_subdir = Str

    def read_data(self):
        data_path = os.path.join(self.master_dir, self.channel_subdir, "point_data.csv")
        if os.path.isfile(data_path):
            return np.loadtxt(data_path, delimiter=",")
        return None

class LineDataIC(IC):
    channel_subdir = Str

    def read_data(self):
        data_path = os.path.join(self.master_dir, self.channel_subdir, "line_data.csv")
        if os.path.isfile(data_path):
            return np.loadtxt(data_path, delimiter=",")
        return None
