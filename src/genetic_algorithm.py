import numpy as np
import random
from .instance import Solution   
import copy   
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import json
import pandas as pd
import heapq

def prim_dijkstra_nx(G, root, beta=0.3, weight='dist'):
    """
    Computes a Shallow-Light Tree (PD-Tree) from a NetworkX graph.
    
    Parameters:
    - G: NetworkX Graph (can be complete or sparse).
    - root: The node identifier for the root (e.g., 0).
    - beta: 0.0 (MST) to 1.0 (SPT/Dijkstra).
    - weight: The edge attribute name holding the distance/cost.
    
    Returns:
    - T: A new NetworkX Graph representing the simplified tree.
    """
    # The result tree
    T = nx.Graph()
    T.add_node(root, **G.nodes[root]) # Copy root attributes if any
    
    # Track visited nodes to avoid cycles
    visited = set()
    
    # Track the actual distance from root to node u in the tree T
    # dist_in_tree[root] = 0
    dist_in_tree = {root: 0.0}
    
    # Priority Queue: stores tuples of (heuristic_cost, u, v, edge_weight)
    # We start by adding all valid edges from the root to the PQ
    # heuristic = weight(u,v) + beta * dist_in_tree[u]
    pq = []
    
    visited.add(root)
    
    # Initialize PQ with root's neighbors
    for neighbor, attr in G[root].items():
        w = attr.get(weight, 1.0)
        # For the root, dist_in_tree[root] is 0, so heuristic is just w
        heapq.heappush(pq, (w, root, neighbor, w))

    # Main Loop
    while pq:
        # Get the edge with the lowest "PD-cost"
        prio, u, v, w = heapq.heappop(pq)
        
        if v in visited:
            continue
            
        # Add v to the tree
        visited.add(v)
        T.add_node(v, **G.nodes[v]) # Copy node attributes
        T.add_edge(u, v, dist=w)
        
        # Calculate exact distance to v in the new tree
        dist_v = dist_in_tree[u] + w
        dist_in_tree[v] = dist_v
        
        # Scan neighbors of v to add to PQ
        for neighbor, attr in G[v].items():
            if neighbor not in visited:
                w_new = attr.get(weight, 1.0)
                
                # THE CORE EQUATION:
                # Cost = Edge_Weight + (Beta * Distance_From_Root_To_Parent)
                heuristic = w_new + (beta * dist_v)
                
                heapq.heappush(pq, (heuristic, v, neighbor, w_new))
                
    return T

class GeneticAlgorithm:
    def __init__(self, problem, population_size=100, max_generations = 500, pd_param=0.5, mutation_rate = 0.3, mutation_choice = 0.5):
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.G = problem.graph.copy()
        self.G = prim_dijkstra_nx(self.G, 0, beta=pd_param)
        naive_solution = Solution(P=problem, G=self.G, solution=[dest for dest in range(1, self.G.number_of_nodes())])
        self.population : list[Solution] = [Solution(P=problem, G=self.G) for _ in range(population_size)] + [naive_solution]
        self.best_cost : float = np.inf
        self.best_sol : Solution = None
        self.mutation_rate = mutation_rate
        self.mutation_choice = mutation_choice
        self.elitism = population_size // 20
        self.history = []
        self.evaluations = []

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
            "baseline_cost": baseline_cost,
            "best_cost": self.best_cost,
        }
        np.save(f"{log_dir}/best_sol.npy", self.best_sol.solution)
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