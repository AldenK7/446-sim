import random
import math
import numpy as np

random.seed(10)

pop_size = 10
resource_size = 3

lambda_a = 10

class Resource:
    def __init__(self, loc, fullness):
        self.loc = loc
        self.fullness = fullness

class Arrival:
    def __init__(self, loc, time):
        self.loc = loc
        self.time = time

    def __str__(self):
        return(
            "Location: ({}, {})\n".format(self.loc[0], self.loc[1]) +
            "Time: {}".format(self.time)
        )

# Generate arrivals
inter_arrivals = [0] * pop_size
arrival_locations = [0] * pop_size
for i in range(pop_size):
    inter_arrivals[i] = -(1/lambda_a) * np.log(1 - random.uniform(0, 1))
    arrival_locations[i] = random.uniform(0, 1)

arrivals = [None] * pop_size

arrivals[0] = Arrival(
    (arrival_locations[0], 0),
    inter_arrivals[0]
)

for i in range(1, pop_size):    
    arrivals[i] = Arrival(
        loc = (arrival_locations[i], 0),
        time = inter_arrivals[i] + arrivals[i-1].time
    )

# Generate enviroment resources
resources = [None] * resource_size
for i in range(resource_size):
    resources[i] = Resource(
        loc = (random.uniform(0, 1), random.uniform(0, 1)),
        fullness = 100
    )

# Main sim loop
for i in range(pop_size):
    arrival = arrivals[i]
    print(arrival)
