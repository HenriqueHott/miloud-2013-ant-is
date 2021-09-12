import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from typing import *
import  random

def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)


def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:
    visibilities = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if i != j:
                x = distances[i, j]
                y = 1 / distances[i, j]
                visibilities[i, j] = 1 / distances[i, j]

    return visibilities


def create_colony(num_ants):
    return np.full((num_ants, num_ants), -1)


def create_pheromone_trails(search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
    trails = np.full(search_space.shape, initial_pheromone)
    np.fill_diagonal(trails, 0)
    return trails


def get_probabilities_paths_ordered(ant: np.array, visibility_rates: np.array, phe_trails) \
        -> Tuple[Tuple[int, Any]]:
    available_instances = np.nonzero(ant < 0)[0]
    # The pheromones over the available paths
    smell = np.sum(
        phe_trails[available_instances]
        * visibility_rates[available_instances])

    # Calculate the probalilty by avaible instance using
    # the sum of pheromones in rest of tour
    probabilities = np.zeros((len(available_instances), 2))
    for i, available_instance in enumerate(available_instances):
        probabilities[i, 0] = available_instance
        path_smell = phe_trails[available_instance] * \
            visibility_rates[available_instance]
        probabilities[i, 1] = path_smell / smell

    sorted_probabilities = probabilities[probabilities[:, 1].argsort()][::-1]
    return tuple([(int(i[0]), i[1]) for i in sorted_probabilities])


def run_colony(X, Y, initial_pheromone, evaporarion_rate, Q):
    distances = get_pairwise_distance(X)
    visibility_rates = get_visibility_rates_by_distances(distances)
    the_colony = create_colony(X.shape[0])
    for i in range(X.shape[0]):
        the_colony[i, i] = 1

    # the_colony[0, 1] = 0
    # the_colony[0, 2] = 0
    # the_colony[1, 0] = 0
    # the_colony[2, 0] = 0
    # the_colony[2, 1] = 0

    last_choices = np.arange(the_colony.shape[0])
    pheromone_trails = create_pheromone_trails(distances, initial_pheromone)

    while -1 in the_colony:
        # Each ant will choose thier next istance
        for i, ant in enumerate(the_colony):
            if -1 in ant:
                ant_pos = last_choices[i]
                choices = get_probabilities_paths_ordered(
                    ant,
                    visibility_rates[ant_pos, :],
                    pheromone_trails[ant_pos, :])

                for choice in choices:
                    next_instance = choice[0]
                    probability = choice[1]
                    if probability == 0:
                        continue

                    ajk = random.randint(0, 1)
                    final_probability = probability * ajk
                    if final_probability != 0:
                        last_choices[i] = next_instance
                        the_colony[i, next_instance] = 1
                        break
                    else:
                        the_colony[i, next_instance] = 0


    print('aaa')


def main():
    dataframe = pd.read_csv("databases/ecoli.csv", header=None)
    last_row = len(dataframe.columns) - 1
    classes = dataframe[last_row]
    dataframe = dataframe.drop(columns=[0, last_row])
    num_instances = len(dataframe.index)
    initial_pheromone = 1
    Q = 1
    evaporation_rate = 0.1
    print('AAAA')
    run_colony(dataframe.to_numpy()[0:4, :], classes,
               initial_pheromone, evaporation_rate, Q)


if __name__ == '__main__':
    main()
