import networkx as nx
import numpy as np
from collections import defaultdict

class Solution:
    def __init__(self, G=None, paths_dict=None, gold_dict=None, solution=None):
        if G is not None:
            self.paths_dict = nx.shortest_path(G, source=0, weight='dist')
            self.gold_dict = {n: data['gold'] for n, data in G.nodes(data=True)}
        elif paths_dict is not None and gold_dict is not None:
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
    
    def mutate(self, verbose=False):
        solution_copy = self.solution.copy()
        idx = np.random.choice(list(self.admissible_mutations.keys())) - 1
        choices = self.admissible_values[idx + 1].copy()
        choices.remove(self.solution[idx])
        c = np.random.choice(choices)
        solution_copy[idx] = c.item()
        if verbose:
            print(f"Mutated index {idx} from {self.solution[idx]} to {c.item()}")
        return Solution(paths_dict=self.paths_dict, gold_dict=self.gold_dict, solution=solution_copy)
    
    def __str__(self):
        return str(self.solution)