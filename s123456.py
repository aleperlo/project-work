from Problem import Problem
from src.genetic_algorithm import GeneticAlgorithm
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

def solution(p:Problem):
    num_cities = p.graph.number_of_nodes()
    G = p.graph
    population_size = 100 if num_cities == 100 else 50
    max_generations = 50 if num_cities == 100 else 25
    density = nx.density(G)
    pd_param = 0.8 if density < 0.5 else 0.5
    GA = GeneticAlgorithm(p, population_size=population_size, max_generations=max_generations, mutation_rate=1, mutation_choice=0.5, pd_param=pd_param)
    best_solution, best_cost = GA.solve()
    path = best_solution.format_solution()
    print(best_cost)
    return path, best_solution

def check_solution(sol: list[tuple[int, float]], p):
    # check a direct edge exists between consecutive cities
    for i in range(len(sol)-1):
        if not p.graph.has_edge(sol[i][0], sol[i+1][0]):
            return "No edge between {} and {}".format(sol[i][0], sol[i+1][0])
    #check gold is taken from all cities
    gold_collected = set(city for city, gold in sol if gold > 0)
    all_cities = set(range(1, p.graph.number_of_nodes()))  
    values = set(best_solution.solution)
    print(f"all gold collected" if gold_collected == all_cities else f"Not all cities collected from. Collected from {gold_collected}, expected {all_cities}")
    # check solution values are correct
    ok = True
    for v in values:
        if v != best_solution.solution[v-1]:
            ok = False
            print("Value mismatch at city {}: expected {}, got {}".format(v,  v, best_solution.solution[v-1]))
            break
    if ok:
        print("Solution values correct")
    # check if gold is collected more than once from any city
    collected_counts = {}
    for city, gold in sol:
        if gold > 0:
            if city in collected_counts:
                collected_counts[city] += 1
            else:
                collected_counts[city] = 1

p = Problem(num_cities=100, alpha=1, beta=2, density=1, seed=np.random.randint(0, 10000))
sol, best_solution = solution(p)
print(p.baseline())
check_solution(sol,p)
