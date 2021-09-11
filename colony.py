from typing import List
from ant import Ant
import numpy as np


class Colony:
    def __init__(
            self,
            adj_matrix: np.ndarray,
            heuristic_values: np.ndarray,
            num_ants: int,
            initial_positions: List[int],
            initial_pheromone: float,
            evaporation_rate: float,
            deposit_factor: float) -> None:

        self.adj_matrix = adj_matrix,
        self.heuristic_values = heuristic_values
        self.cities = adj_matrix.shape[0]
        self.initial_pheromone = initial_pheromone
        self.num_ants = num_ants
        self.initial_positions = initial_positions
        self.evaporation_rate = evaporation_rate
        self.deposit_scale = deposit_factor

        self.ants = [Ant(self.initial_positions[i], self.cities)
                     for i in range(self.num_ants)]

        self.pheromones = np.full(
            (self.cities, self.cities), self.initial_pheromone, dtype=np.float64)

        np.fill_diagonal(self.pheromones, 0.0)

    def update_pheromones(self):
        adj_matrix = self.adj_matrix[0]
        pheromone_deposited = np.zeros(self.pheromones.shape)
        for ant in self.ants:
            tour_lenght = 0
            for i, j in ant.path_done:
                tour_lenght += adj_matrix[i, j]

            if tour_lenght <= 0:
                continue

            for i, j in ant.path_done:
                pheromone_deposited[i, j] += self.deposit_scale / tour_lenght

        for i in range(self.cities):
            for j in range(self.cities):
                if i == j:
                    continue

                rate = 1 - self.evaporation_rate
                pheromone = self.pheromones[i, j]
                self.pheromones[i, j] = \
                    (rate * pheromone) \
                    + pheromone_deposited[i, j]

    @property
    def colony_finished(self) -> bool:
        for ant in self.ants:
            if not ant.tour_completed:
                return False

        return True

    def run_colony(self) -> None:
        while True:
            for ant in self.ants:
                ant.choose_next_destination(
                    self.pheromones[ant.last_pos, :], self.heuristic_values[ant.last_pos, :])

                self.update_pheromones()

            if self.colony_finished:
                break

    @property
    def solutions(self) -> np.ndarray:
        return np.array([ant.choices for ant in self.ants])
