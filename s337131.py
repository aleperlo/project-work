from Problem import Problem
from src.genetic_algorithm import GeneticAlgorithm
import networkx as nx
import random
import numpy as np

def solution(p:Problem):
    random.seed(42)
    np.random.seed(42)
    num_cities = p.graph.number_of_nodes()
    G = p.graph
    population_size = 100 if num_cities <= 500 else 50
    max_generations = 50 if num_cities <= 500 else 25
    density = nx.density(G)
    pd_param = 0.8 if density < 0.5 else 0.5
    GA = GeneticAlgorithm(p, population_size=population_size, max_generations=max_generations, mutation_rate=0.5, mutation_choice=0.5, pd_param=pd_param)
    best_solution, _ = GA.solve()
    path = best_solution.format_solution()
    return path