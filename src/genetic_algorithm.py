import numpy as np
import random
from .instance import Solution      

class GeneticAlgorithm:
    def __init__(self, problem, population_size=50, max_generations = 200, paths_dict=None, gold_dict=None):
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.population = [Solution(P=problem, paths_dict=paths_dict, gold_dict=gold_dict) for _ in range(population_size)]
        self.best_cost = np.inf
        self.best_sol = None

    def tournament_selection(self, costs, tao = 3):
        indeces = random.sample(range(self.population_size), tao)
        idx = min(indeces, key=lambda i: costs[i])
        return self.population[idx]

    def solve(self):
        for _ in range(self.max_generations):
            costs = [ind.fitness() for ind in self.population]
            min_cost = min(costs)
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                best_idx = costs.index(min_cost) 
                self.best_sol = self.population[best_idx]

            next_gen = [self.best_sol]
            
            while len(next_gen) < self.population_size:
                p = self.tournament_selection(costs)
                offspring = p.mutate()
                next_gen.append(offspring)
            
            self.population = next_gen
            
        return self.best_sol.formatted_solution, self.best_cost