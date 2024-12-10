from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from  matplotlib.animation import FuncAnimation

# Map creation ----------------------------------------------------------------
class SeaMap:
    def __init__(self, shape, frequency, volume_mean, volume_sd, nest, save_map):
        self.shape = shape
        self.frequency = frequency 
        self.volume_mean = volume_mean
        self.volume_sd = volume_sd
        self.nest = nest

        self.map = np.zeros(shape)

        self.map_history = []
        self.save_map = save_map

    # Save map state for animation
    def __save_map(self):
        self.map_history.append(np.array(self.map))

    # Generate resource point (point = uniform() : volume = normal())
    def __gen_resource_point(self, x, y):
        if random.uniform(0, 1) < self.frequency and (x != self.nest[0] and y != self.nest[1]):
            volume = max(0, random.normal(loc=self.volume_mean, scale=self.volume_sd))
            self.map[x][y] = volume
        else:
            self.map[x][y] = 0

    # Get total volume in the map
    def get_total_volume(self):
        return np.sum(self.map)

    # Consume volume in that point, return amount consumed
    def consume_point(self, x, y, val):
        consumed = 0
        if self.map[x][y] - val >= 0:
            consumed = val
            self.map[x][y] = self.map[x][y] - val
        else:
            consumed = self.map[x][y]
            self.map[x][y] = 0

        if self.save_map: self.__save_map()
        return consumed

    # Fill initial points
    def gen_map(self):        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.__gen_resource_point(i, j)

        if self.save_map: self.__save_map()

    # Shift map
    def shift_map(self, shiftx, shifty):
        self.map = shift(self.map, (shiftx, shifty), cval=np.nan)

        nans = np.argwhere(np.isnan(self.map))

        for coord in nans:
            self.__gen_resource_point(coord[0], coord[1])
        
        if self.save_map: self.__save_map()

    # Map visualization
    def show_map(self):
        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(self.map, cmap="seismic")

        for (i, j), z in np.ndenumerate(self.map):
            if z > 1e-10:
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

        plt.show()

    def animate_map(self, speed):
        fig = plt.figure()
        plot = plt.imshow(self.map_history[0])

        def update(i):
            plot.set_data(self.map_history[i * speed])
            return [plot]

        anim = FuncAnimation(fig, update, interval=1, frames=len(self.map_history) // speed, blit=True)
        plt.show()



# Event -----------------------------------------------------------------------
import heapq as hq

class Event:
    ARRIVAL = 0
    DEPARTURE = 1
    MAP = 2

    def __init__(self, id, type, time, shift=(0,0), coord=(-1,-1)):
        self.id = id
        self.type = type
        self.time = time
        self.shift = shift
        self.coord = coord

    def __lt__(self, other):
        return (self.time < other.time)



# Sim -------------------------------------------------------------------------
class Stats:
    def __init__(self):
        self.consumptions = []
        self.distances = []
        self.volumes = []

        self.consumptions_t = []
        self.distances_t = []
        self.volumes_t = []

        self.low_consumption = []
        self.low_consumption_t = []

    def add_consumption(self, tuple):
        # If consumption is 3 sd's away from mean
        if tuple[1] < 0.17 - (3 * 0.04):
            self.low_consumption.append(tuple[1])
            self.low_consumption_t.append(tuple[0])
        
        self.consumptions.append(tuple[1])
        self.consumptions_t.append(tuple[0])
    
    def add_distance(self, tuple):
        self.distances.append(tuple[1])
        self.distances_t.append(tuple[0])
    
    def add_volume(self, tuple):
        self.volumes.append(tuple[1])
        self.volumes_t.append(tuple[0])

class Sim:
    # Time in hours
    # Volume in kg

    DAY = 24
    SHIFT_RANGE = (1, 3)
    
    # We found an inter arrival rate of 2.4 hrs with a sample of 129 penguins.
    # Adelie Land has ~36000 penguins, so scaling inter arrival rate
    ARRIVAL_RATE = (2.4216 / (36000 / 129)) # From our input modelling

    def __init__(self, nest_location, grid_shape, resource_frequency, 
                 volume_mean, total_days, animate=False):
        # Parameters for map
        self.nest_location = nest_location
        self.grid_shape = grid_shape
        self.resource_frequency = resource_frequency
        self.volume_mean = volume_mean
        self.volume_sd = 0.1 * volume_mean
        self.total_days = total_days

        self.seamap = SeaMap(
            shape = self.grid_shape, 
            frequency = self.resource_frequency,
            volume_mean = self.volume_mean,
            volume_sd = self.volume_sd,
            nest = self.nest_location,
            save_map = animate,
        )

        self.clock = 0
        self.event_list = []
        self.event_count = 0

        self.stats = Stats()

    def reset(self):
        self.seamap = SeaMap(
            shape = self.grid_shape, 
            frequency = self.resource_frequency,
            volume_mean = self.volume_mean,
            volume_sd = self.volume_sd,
            nest = self.nest_location,
            save_map = self.seamap.save_map,
        )

        self.clock = 0
        self.event_list = []
        self.event_count = 0

        self.stats = Stats()

    def initialize_sim(self):
        # Generate map
        self.seamap.gen_map()

        # Create first arrival
        arrival_event = Event(
            self.event_count, Event.ARRIVAL, 
            self.clock + random.exponential(self.ARRIVAL_RATE)
        )
        self.event_count += 1
        hq.heappush(self.event_list, arrival_event)

        # Create first map event
        rand_shift = (
            -random.randint(self.SHIFT_RANGE[0], self.SHIFT_RANGE[1]),
            random.randint(self.SHIFT_RANGE[0], self.SHIFT_RANGE[1])
        )
        map_event = Event(self.event_count, Event.MAP, self.clock + self.DAY, shift=rand_shift)
        self.event_count += 1
        hq.heappush(self.event_list, map_event)
    
    def __square_distance(self, point1, point2):
        return(
            np.power(point1[0]-point2[0], 2) +
            np.power(point1[1]-point2[1], 2)
        )

    def process_map_shift(self, event):
        self.seamap.shift_map(event.shift[0], event.shift[1])

        # Create another map event
        rand_shift = (
            -random.randint(self.SHIFT_RANGE[0], self.SHIFT_RANGE[1]),
            random.randint(self.SHIFT_RANGE[0], self.SHIFT_RANGE[1])
        )
        map_event = Event(self.event_count, Event.MAP, self.clock + self.DAY, shift=rand_shift)
        self.event_count += 1
        hq.heappush(self.event_list, map_event)

    def process_arrival(self, event):
        # Get all resource locations
        coords = np.transpose(np.nonzero(self.seamap.map))

        # No resource points
        if len(coords) <= 0:
            # Do stats
            self.stats.add_consumption([self.clock, 0]) 
            self.stats.add_distance([self.clock, -1])
            self.stats.add_volume([self.clock, self.seamap.get_total_volume()])

            # Create another arrival
            arrival_event = Event(
                self.event_count, Event.ARRIVAL, 
                self.clock + random.exponential(self.ARRIVAL_RATE)
            )
            self.event_count += 1
            hq.heappush(self.event_list, arrival_event)

            return

        # Calc distance and get volumes
        distances = np.zeros(coords.shape[0])
        volumes = np.zeros(coords.shape[0])
        for i in range(coords.shape[0]):
            distances[i] = self.__square_distance(self.nest_location, coords[i])
            volumes[i] = self.seamap.map[coords[i][0]][coords[i][1]]

        # Normalize and combine to create rating
        if np.max(distances) == np.min(distances):
            # Nothing to normalize, just make array of 1
            distances = np.ones(distances.shape[0]) 
            volumes = np.array(volumes.shape[0])
        else:
            distances = 1 - ((distances-np.min(distances)) / (np.max(distances)-np.min(distances))) 
            volumes = (volumes-np.min(volumes)) / (np.max(volumes)-np.min(volumes)) 

        ratings = (distances + volumes) / 2
        sort_ind = np.argsort(ratings)[::-1]

        # Get all high ratings based on an epsilon
        epsilon = 0.1
        cur_rating = ratings[sort_ind[0]]
        high_rated = [sort_ind[0]]
        i = 0 

        while(cur_rating >= ratings[sort_ind[0]] - epsilon and i < len(ratings) - 1):
            high_rated.append(sort_ind[i])
            i += 1
            cur_rating = ratings[sort_ind[i]]

        # Pick at random
        choice = high_rated[random.choice(len(high_rated), 1)[0]]

        # Consume
        consumption = random.normal(0.17, 0.04) # Distribution from https://www.nature.com/articles/s41598-021-02451-4#Sec9
        true_consumption = self.seamap.consume_point(coords[choice][0], coords[choice][1], consumption)

        # Do stats
        self.stats.add_consumption([self.clock, true_consumption]) 
        self.stats.add_distance([
            self.clock, 
            np.sqrt(self.__square_distance(self.nest_location, coords[choice]))
        ])
        self.stats.add_volume([self.clock, self.seamap.get_total_volume()])

        # Create another arrival
        arrival_event = Event(
            self.event_count, Event.ARRIVAL, 
            self.clock + random.exponential(self.ARRIVAL_RATE)
        )
        self.event_count += 1
        hq.heappush(self.event_list, arrival_event)

    def run(self):
        self.initialize_sim()

        # Main loop
        while len(self.event_list) > 0 and self.clock < self.DAY * self.total_days:
            cur_event = hq.heappop(self.event_list)
            self.clock = cur_event.time

            if cur_event.type == Event.ARRIVAL:
                self.process_arrival(cur_event)
            else:
                # print("Day", self.clock // self.DAY, "...")
                self.process_map_shift(cur_event)
        
        self.stats.consumptions = np.array(self.stats.consumptions)
        self.stats.distances = np.array(self.stats.distances)
        self.stats.volumes = np.array(self.stats.volumes)




# Sim Runs --------------------------------------------------------------------

tests = [
    # Baseline
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5,
        volume_mean=100,
        total_days=90,
        animate=True
    ),
    "dir":"baseline"
    },

    # 75% volume
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5,
        volume_mean=100 * 0.75,
        total_days=90,
        animate=True
    ),
    "dir":"75volume"
    },

    # 50% volume
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5,
        volume_mean=100 * 0.50,
        total_days=90,
        animate=True
    ),
    "dir":"50volume"
    },

    # 25% volume
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5,
        volume_mean=100 * 0.25,
        total_days=90,
        animate=True
    ),
    "dir":"25volume"
    },

    # 75% frequency
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5 * 0.75,
        volume_mean=100,
        total_days=90,
        animate=True
    ),
    "dir":"75frequency"
    },

    # 50% frequency
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5 * 0.50,
        volume_mean=100,
        total_days=90,
        animate=True
    ),
    "dir":"50frequency"
    },

    # 25% frequency
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5 * 0.25,
        volume_mean=100,
        total_days=90,
        animate=True
    ),
    "dir":"25frequency"
    },

    # 75% volume and frequency
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5 * 0.75,
        volume_mean=100 * 0.75,
        total_days=90,
        animate=True
    ),
    "dir":"75vol75freq"
    },

    # 50% volume and frequency
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5 * 0.50,
        volume_mean=100 * 0.50,
        total_days=90,
        animate=True
    ),
    "dir":"50vol50freq"
    },

    # 25% volume and frequency
    {"sim": Sim(
        nest_location=(0,0),
        grid_shape=(10,10),
        resource_frequency=0.5 * 0.25,
        volume_mean=100 * 0.25,
        total_days=90,
        animate=True
    ),
    "dir":"25vol25freq"
    },
]

for test in tests:
    sim = test["sim"]
    dir = test["dir"]

    print("----", dir, "----")

    for i in range(1, 6):
        seed = i
        random.seed(seed)

        print("Run:", seed)
        sim.reset()
        sim.run()

        print(np.mean(sim.stats.consumptions))
        print(np.mean(sim.stats.distances))
        print(len(sim.stats.low_consumption) / len(sim.stats.consumptions))

        seed_arr = np.full(sim.stats.consumptions.shape, seed)

        combined_stats = np.dstack((sim.stats.consumptions_t, sim.stats.consumptions, sim.stats.distances, sim.stats.volumes, seed_arr))[0, :, :]
        print(combined_stats.shape)

        np.savetxt(
            "sim-data/"+dir+"/"+str(seed)+".csv", combined_stats, delimiter=",", 
            header="time,consumption,distance,volume,seed", comments=''
        )

# sim.seamap.animate_map(speed=10)