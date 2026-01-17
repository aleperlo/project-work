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
    GA = GeneticAlgorithm(p, population_size=population_size, max_generations=max_generations, mutation_rate=0, mutation_choice=0.5, pd_param=pd_param)
    best_solution, best_cost = GA.solve()
    path = best_solution.format_solution()
    print(best_solution)
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
    ok = True
    for v in values:
        if v != best_solution.solution[v-1]:
            ok = False
            break
    print("Solution values correct:", ok)
    return f"ok" if gold_collected == all_cities else f"Not all cities collected from. Collected from {gold_collected}, expected {all_cities}"

p = Problem(num_cities=10, alpha=1, beta=1, density=0.2, seed=random.randint(0, 10000))
sol, best_solution = solution(p)
print(sol)
print("Solution valid:", check_solution(sol, p))
