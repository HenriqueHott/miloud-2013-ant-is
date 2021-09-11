import random
from typing import List, Tuple
import numpy as np


class Ant:
    def __init__(self, intial_pos: int, num_cities: int) -> None:
        self.num_cites = num_cities
        self.choices = np.full(num_cities, -1)
        self.tour = [intial_pos]
        self.choices[intial_pos] = 1
        self.last_pos = intial_pos

    @property
    def tour_completed(self) -> bool:
        return -1 not in self.choices

    @property
    def path_done(self) -> List[Tuple[int, int]]:
        path = []
        for i in range(len(self.tour) - 1):
            path.append((self.tour[i], self.tour[i + 1]))

        return path

    def get_available_cities(self) -> np.array:
        return np.argwhere(self.choices < 0).flatten()

    def choose_next_destination(
            self,
            pheromones: np.ndarray,
            heuristic_values: np.ndarray,):
        if self.tour_completed:
            return

        available_cities = self.get_available_cities()
        trail_smells = np.sum(
            pheromones[available_cities]
            * heuristic_values[available_cities])

        probs = np.zeros((self.num_cites, 2))
        for city in available_cities:
            probs[city, 0] = city
            path_smell = pheromones[city] * heuristic_values[city]

            probs[city, 1] = path_smell / trail_smells

        probs = probs[probs[:, 1].argsort()][::-1]
        for new_city, prob in probs:
            new_city = int(new_city)
            if prob == 0:
                self.choices[new_city] = 0
                continue

            fprob = prob * random.randint(0, 1)
            if fprob != 0:
                self.choices[new_city] = 1
                self.last_pos = new_city
                self.tour.append(new_city)
                break
            else:
                self.choices[new_city] = 0
