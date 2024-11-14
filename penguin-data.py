import pandas as pd
import numpy as np
from geopy import distance
from matplotlib import pyplot as plt 

pd.options.mode.chained_assignment = None

# Read in CSV and do some preprocessing
df = pd.read_csv("penguin.csv")
df.dropna(inplace=True)
df.drop(columns=[
    "event-id", 
    "visible", 
    "migration-stage", 
    "tag-tech-spec", 
    "sensor-type",
    "individual-local-identifier",
    "study-name",
    "individual-taxon-canonical-name"
    ], inplace=True)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Seperate data by season
seasons = df["comments"].unique().tolist()

# Define nest location and boundary
nest_location = (-66.663596, 140.004067)
radius = 1.0 # 1km away from nest

departures = {} # Departure timestamps per season
for season in seasons:
    season_df = df[df["comments"] == season]
    penguin_ids = season_df["tag-local-identifier"].unique().tolist()

    season_departures = []

    # Look through all penguins 
    for id in penguin_ids:
        penguin_df = season_df[season_df["tag-local-identifier"] == id]
        penguin_df.sort_values(by="timestamp", inplace=True)

        coord = (penguin_df["location-lat"].iloc[0], penguin_df["location-long"].iloc[0])
        prev_dist = distance.distance(nest_location, coord).km

        # Look through all movement lines
        for i in range(1, penguin_df.shape[0]):
            coord = (penguin_df["location-lat"].iloc[i], penguin_df["location-long"].iloc[i])
            cur_dist = distance.distance(nest_location, coord).km

            # If moved out of radius, add timestamp
            if prev_dist < radius and cur_dist >= radius:
                season_departures.append(penguin_df["timestamp"].iloc[i])

            prev_dist = cur_dist

# TODO Histogram stuff
# season_departures = np.array(season_departures)

# season_departures = np.sort(season_departures)

# to_timestamp = np.vectorize(lambda x: x.timestamp())
# time_stamps = to_timestamp(season_departures)
# values, bins = np.histogram(time_stamps, bins=10)
# print(values)
# print(bins)

# plt.hist(time_stamps, bins=bins) 
# plt.title("histogram") 
# plt.show()

# departures[season] = season_departures
# print(departures[season].shape[0])
