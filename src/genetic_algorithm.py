import numpy as np
import random
from .instance import Solution   
import copy   
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import json
import pandas as pd

class GeneticAlgorithm:
    def __init__(self, problem, population_size=100, max_generations = 500, disconnection_ratio=0.7, mutation_rate = 0.3, mutation_choice = 0.5):
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.paths_dict = nx.shortest_path(problem.graph, source=0, weight='dist')
        self.G = problem.graph.copy()
        self.paths_dict.pop(0)
        if nx.density(self.G) > 0.7: # if the graph is almost complete, disconnect
                self.disconnected_graph(ratio=disconnection_ratio)
        self.population : list[Solution] = [Solution(P=problem, G=self.G) for _ in range(population_size)]
        self.best_cost : float = np.inf
        self.best_sol : Solution = None
        self.mutation_rate = mutation_rate
        self.mutation_choice = mutation_choice
        self.elitism = self.population_size // 10
        self.history = []
        self.evaluations = []

    def disconnected_graph(self, ratio):    
        # Remove the starting node from paths_dict
        dists = [self.G[0][i]["dist"] for i in self.paths_dict.keys()]
        dists = dists / sum(dists)
        to_remove = random.choices(list(self.paths_dict.keys()), weights=dists, k=int(ratio*len(self.paths_dict)))
        #print(f"Removing edges from 0 to nodes: {to_remove}")
        self.G.remove_edges_from([(0, node) for node in to_remove])
        while nx.density(self.G) > 0.3:
            u, v = random.choice(list(self.G.edges()))
            if u != 0 and v != 0:
                self.G.remove_edge(u, v)

    def tournament_selection(self, costs, tao = 3):
        indeces = random.sample(range(self.population_size), tao)
        idx = min(indeces, key=lambda i: costs[i])
        return self.population[idx]

    def solve(self):
        initial_costs = [ind.fitness() for ind in self.population]
        best_idx = initial_costs.index(min(initial_costs))
        self.best_sol = self.population[best_idx]
        self.best_cost = initial_costs[best_idx]
        
        for i in tqdm(range(self.max_generations)):
            costs = [ind.fitness() for ind in self.population]
            min_costs = np.argsort(costs)[:self.elitism]
            min_cost = costs[min_costs[0]]
            self.history.append((i, min_cost))
            gen_evaluations = []

            if min_cost < self.best_cost:
                self.best_cost = min_cost
                best_idx = min_costs[0]
                self.best_sol = self.population[best_idx]

            next_gen = [copy.deepcopy(self.population[idx]) for idx in min_costs]
            
            while len(next_gen) < self.population_size:
                p1 = self.tournament_selection(costs)                

                if random.random() < self.mutation_rate:
                    if random.random() < self.mutation_choice:
                        offspring = p1.mutate_split()
                        
                    else:
                        offspring = p1.mutate_join()
                       
                else:
                    p2 = self.tournament_selection(costs)
                    offspring = p1.crossover(p2)

                gen_evaluations.append(offspring.fitness())

                next_gen.append(offspring)
            self.evaluations.append((i, gen_evaluations))
            
            self.population = next_gen
            
        return self.best_sol, self.best_cost

    def plot_history(self):
        cumulative_avg = np.cumsum([cost for _, cost in self.history]) / np.arange(1, len(self.history)+1)
        generations = [gen for gen, _ in self.history]        
        plt.plot(generations, cumulative_avg)
        plt.xlabel('Generation')
        plt.ylabel('Average Best Cost')
        plt.title('Genetic Algorithm Progress')
        # cloud of points for evaluations
        for gen, evals in self.evaluations:
            plt.scatter([gen]*len(evals), evals, color='red', alpha=0.1, s=1)
        plt.show()

    
    def log(self, log_dir, baseline_cost = None):
        params = {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "mutation_rate": self.mutation_rate,
            "mutation_choice": self.mutation_choice,
            "problem_size": self.G.number_of_nodes(),
            "problem_alpha": self.problem.alpha,
            "problem_beta": self.problem.beta,
            "problem_density": nx.density(self.problem.graph),
            "baseline_cost": baseline_cost
        }
        with open(f"{log_dir}/ga_params.json", "w") as f:
            json.dump(params, f, indent=4)
        df_history = pd.DataFrame(self.history, columns=["generation", "best_cost"])
        df_history.to_csv(f"{log_dir}/ga_history.csv", index=False)
        evaluation_list = []
        for gen, evals in self.evaluations:
            for eval in evals:
                evaluation_list.append((gen, eval))
        df_evaluations = pd.DataFrame(evaluation_list, columns=["generation", "evaluation_cost"])
        df_evaluations.to_csv(f"{log_dir}/ga_evaluations.csv", index=False)