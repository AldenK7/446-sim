from numpy import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift

random.seed(10)

# Map creation ----------------------------------------------------------------
nest_location = (0, 0)
grid_shape = (10, 10)

# Parameters for map
resource_frequency = 0.2
volume_mean = 10
volume_sd = 3

class ResourcePoint:
    def __init__(self, x, y, volume):
        self.x = x
        self.y = y
        self.volume = volume
    
    def __str__(self):
        string = "({}, {}) - {}".format(self.x, self.y, self.volume)
        return string
    
class SeaMap:
    def __init__(self, shape, frequency, volume_mean, volume_sd, nest):
        self.shape = shape
        self.frequency = frequency 
        self.volume_mean = volume_mean
        self.volume_sd = volume_sd
        self.nest = nest

        self.map = np.zeros(shape)
        self.resource_points = []

    def gen_map(self):        
        # Determine interest point locations and volumes
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if random.uniform(0, 1) < self.frequency and (i, j) != self.nest:
                    volume = max(0, random.normal(loc=self.volume_mean, scale=self.volume_sd))
                    self.map[i][j] = volume
                    self.resource_points.append(ResourcePoint(i, j, volume))

    def update_map(self, shiftx, shifty):
        self.map = shift(self.map, (shiftx, shifty), cval=np.NaN)

    def __str__(self):
        string = "({}, {}) - {}".format(self.x, self.y, self.volume)
        return string


# Arrivals --------------------------------------------------------------------

# Main loop -------------------------------------------------------------------

