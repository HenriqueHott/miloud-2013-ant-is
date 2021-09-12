import numpy as np
from colony import Colony
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

dataset_path = "databases/ecoli.csv"

dataframe = pd.read_csv("databases/ecoli.csv", header=None)
last_row = len(dataframe.columns) - 1
classes = dataframe[last_row]
dataframe = dataframe.drop(columns=[0, last_row])
num_instances = len(dataframe.index)

initial_pheromone = 1
Q = 1
evaporation_rate = 0.1


distances = euclidean_distances(dataframe)
visibilities = np.zeros(distances.shape)
for i in range(num_instances):
    for j in range(num_instances):
        if i != j:
            visibilities[i, j] = 1 / distances[i, j]

the_colony = Colony(
    adj_matrix=distances,
    heuristic_values=visibilities,
    initial_pheromone=initial_pheromone,
    evaporation_rate=evaporation_rate,
    num_ants=num_instances,
    initial_positions=[i for i in range(num_instances)],
    deposit_factor=Q
)
the_colony.run_colony()
print(the_colony.solutions)
