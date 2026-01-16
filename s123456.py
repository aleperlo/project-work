from Problem import Problem
from src.genetic_algorithm import GeneticAlgorithm
import networkx as nx
import matplotlib.pyplot as plt
import random

def solution(p:Problem):
    num_cities = p.graph.number_of_nodes()
    G = p.graph
    population_size = 50 if num_cities == 100 else 20
    max_generations = 50 if num_cities == 100 else 20
    density = nx.density(G)
    pd_param = 0.3 if density < 0.5 else 0.5
    GA = GeneticAlgorithm(p, population_size=population_size, max_generations=max_generations, mutation_rate=0.5, mutation_choice=0.5, pd_param=pd_param)
    best_solution, best_cost = GA.solve()
    path = best_solution.format_solution()
    return path

p = Problem(num_cities=10, alpha=1, beta=1, density=1, seed=random.randint(0,10000))
sol = solution(p)
print(sol)