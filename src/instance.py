import networkx as nx
import numpy as np
from collections import defaultdict

class Solution:
    def __init__(self, P, G=None, paths_dict=None, gold_dict=None, solution=None):
        self.P = P
        if G is not None:
            self.G = G
            self.paths_dict = nx.shortest_path(G, source=0, weight='dist')
            self.gold_dict = {n: data['gold'] for n, data in G.nodes(data=True)}
        elif paths_dict is not None and gold_dict is not None:
            self.G = P.graph  # fallback to P.graph when G not provided
            self.paths_dict = paths_dict
            self.gold_dict = gold_dict
        else:
            raise ValueError("Either G or both paths_dict and gold_dict must be provided.")
        # Compute admissible values
        self.admissible_values = defaultdict(list)
        for dest, path in self.paths_dict.items():
            for node in path:
                self.admissible_values[node].append(dest)
        self.admissible_values.pop(0)
        # Initialize solution
        self.solution = self.random_solution() if solution is None else solution
        # Format solution
        self.formatted_solution = self.format_solution()
        # Compute admissible mutations
        self.admissible_mutations = {k: v for k, v in self.admissible_values.items() if len(v) > 1}

    def random_solution(self):
        solution = []
        for i in range(1, len(self.paths_dict)):
            c = np.random.choice(self.admissible_values[i])
            solution.append(c.item())
        return solution
    
    def format_solution(self):
        formatted_solution = []
        for dest, path in self.paths_dict.items():
            if dest == 0:
                continue
            if dest in self.solution:
                # print("dest:", dest, "path:", path, end=' ')
                nodes_to_grab = [i for i in path[1:] if self.solution[i-1] == dest]
                # print("nodes to grab:", nodes_to_grab)
                for node in path[1:-1]:
                    # print("  node:", node, "grab:", 0.0)
                    formatted_solution.append((node, 0.0))
                for node in path[len(path)-1::-1]:
                    gold = self.gold_dict[node] if node in nodes_to_grab else 0.0
                    # print("  node:", node, "grab:", gold)
                    formatted_solution.append((node, gold))
        return formatted_solution
    
    """
    def mutate(self, verbose=False):
        solution_copy = self.solution.copy()
        idx = np.random.choice(list(self.admissible_mutations.keys())) - 1
        choices = self.admissible_values[idx + 1].copy()
        choices.remove(self.solution[idx])
        c = np.random.choice(choices)
        solution_copy[idx] = c.item()
        return Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, solution=solution_copy)
    """
    def mutate_split(self):
        solution_copy = np.array(self.solution.copy())
        values, cnt = np.unique(solution_copy, return_counts=True)
        val = np.random.choice(values, p= cnt/cnt.sum())
        idx = np.where(values == val)[0][0]
        if cnt[idx] > 1:
            i = cnt[idx]//2
            to_change = []
            for node in self.paths_dict[val]:
                if solution_copy[node-1] == val:
                    to_change.append(node-1)
                if len(to_change) >= i:
                    break
            solution_copy[to_change] = i
        return Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, solution=solution_copy)

    def mutate_join(self):
        solution_copy = np.array(self.solution.copy())
        values, cnt = np.unique(solution_copy, return_counts=True)
        prob = 1/cnt / (1/cnt).sum()
        val = np.random.choice(values, p= prob)
        idx = np.where(values == val)[0][0]
        # from solution_copy, choose an index i whose value in the array != val but such that path_dict[i] contains val
        candidates = [i for i in range(len(solution_copy)) if solution_copy[i] != val and val in self.paths_dict[solution_copy[i]]]
        if len(candidates) > 0:
            i = np.random.choice(candidates)
            solution_copy[solution_copy == val] = i+1
        return Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, solution=solution_copy)
    
    def crossover(self, p2):
        offspring = np.zeros(len(self.solution), dtype=int)
        v1 = set(np.unique(self.solution, return_counts=False))
        v2 = set(np.unique(p2.solution, return_counts=False))
        while np.any(offspring == 0): 
            if len(v1) != 0:          
                val = np.random.choice(list(v1))
                v1.remove(val)            
                offspring[(self.solution == val) & (offspring==0)] = val
            if len(v2) != 0:   
                val = np.random.choice(list(v2))
                v2.remove(val)
                offspring[(p2.solution == val) & (offspring==0)] = val
 
        return Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, solution=offspring)

    
    def fitness(self):
        total_cost = 0.0
        for dest, path in self.paths_dict.items():
            if dest == 0 or dest not in self.solution:
                continue
            for i in range(1, len(path)):
                total_cost += self.G[path[i-1]][path[i]]['dist']
                #print(f"distanze from {path[i-1]} to {path[i]}: {self.G[path[i-1]][path[i]]['dist']}")
            nodes_to_grab = [i for i in path[1:] if self.solution[i-1] == dest]
            current_gold = 0.0
            for i in range(len(path)-1, 0, -1):
                if path[i] in nodes_to_grab:
                    current_gold += self.gold_dict[path[i]]
                dist = self.G[path[i-1]][path[i]]['dist']
                #print(f"Going from {path[i]} to {path[i-1]} with distance {dist} and current gold {current_gold}")
                total_cost += dist + (self.P.alpha * dist * current_gold) ** self.P.beta
               
        return total_cost

    def __str__(self):
        return str(self.solution)