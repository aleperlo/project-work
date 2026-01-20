import networkx as nx
import numpy as np
from collections import defaultdict

class Solution:
    def __init__(self, P, G=None, paths_dict=None, gold_dict=None, orig_paths_dict=None, solution=None):
        self.P = P
        if G is not None:
            self.G = G.copy()
            self.paths_dict = nx.single_source_dijkstra_path(self.G, source=0, weight='dist')
            self.gold_dict = {n: data['gold'] for n, data in self.G.nodes(data=True)}
            self.orig_paths_dict = nx.single_source_dijkstra_path(P.graph, source=0, weight='dist')
            
        elif paths_dict is not None and gold_dict is not None and orig_paths_dict is not None:
            self.G = P.graph  # fallback to P.graph when G not provided
            self.paths_dict = paths_dict
            self.gold_dict = gold_dict
            self.orig_paths_dict = orig_paths_dict
        else:
            raise ValueError("Either G or both paths_dict and gold_dict must be provided.")
        # Compute admissible values
        self.admissible_values = defaultdict(list)
        for dest, path in self.paths_dict.items():
            for node in path:
                self.admissible_values[node].append(dest)
        self.admissible_values.pop(0)
        # Compute admissible mutations
        self.admissible_mutations = {k: v for k, v in self.admissible_values.items() if len(v) > 1}
        # Initialize solution
        self.solution = self.random_solution() if solution is None else solution
        self.fitness_value = None

    def random_solution(self):
        solution = []
        for i in range(1, len(self.paths_dict)):
            c = np.random.choice(self.admissible_values[i])
            solution.append(c.item())
        values = set(solution)
        for v in values:
            if v != solution[v-1]:
                solution[v-1] = v
        return solution
    
    def mutation_random_solution(self):
        solution = np.arange(1, len(self.paths_dict))      
        solution_obj = Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, orig_paths_dict=self.orig_paths_dict, solution=solution)
        for _ in range(self.P.graph.number_of_nodes() // 10):
            solution_obj = solution_obj.mutate_join()            
        return solution_obj.solution

    def format_solution(self):
        formatted_solution = []
        for dest, path in self.paths_dict.items():
            if dest == 0:
                continue
            orig_path = self.orig_paths_dict[dest]
            if dest in self.solution:
                # print("dest:", dest, "path:", path, end=' ')
                nodes_to_grab = [i for i in path[1:] if self.solution[i-1] == dest]
                # node with lowest index in path
                nearest_to_grab = min(nodes_to_grab, key=lambda x: path.index(x))
                #print("nodes to grab:", nodes_to_grab, "nearest to grab:", nearest_to_grab)
                # print("nodes to grab:", nodes_to_grab)
                for node in orig_path[1:-1]:
                    # print("  node:", node, "grab:", 0.0)
                    formatted_solution.append((node, 0.0))
                if len(nodes_to_grab) == 1 and nodes_to_grab[0] == path[-1]:
                    return_path = orig_path

                else:
                    #print("dest", dest, "Nodes to grab:", nodes_to_grab,"path", path, "nearest to grab:", nearest_to_grab)
                    return_path = self.orig_paths_dict[nearest_to_grab] + path[path.index(nearest_to_grab)+1:]
                    #print("Return path:", return_path)
                
                for node in return_path[len(return_path)-1::-1]:
                    gold = self.gold_dict[node] if node in nodes_to_grab else 0.0
                    # print("  node:", node, "grab:", gold)
                    formatted_solution.append((node, gold))
        return formatted_solution
    

    """
    def format_solution(self):
        formatted_solution = []
        for dest, path in self.paths_dict.items():
            if dest == 0:
                continue
            orig_path = self.orig_paths_dict[dest]
            if dest in self.solution:
                nodes_to_grab = [i for i in path[1:] if self.solution[i-1] == dest]
                print("nodes to grab while reaching [", dest, "]", nodes_to_grab)
                for node in orig_path[1:-1]:
                    #print("node:", node, "grab:", 0.0)
                    formatted_solution.append((node, 0.0))
                if len(nodes_to_grab) == 1 and nodes_to_grab[0] == path[-1]:
                    return_path = orig_path
                else:
                    return_path = path
                for node in return_path[len(return_path)-1::-1]:
                    gold = self.gold_dict[node] if node in nodes_to_grab else 0.0
                    #print("  node:", node, "grab:", gold)
                    formatted_solution.append((node, gold))
        return formatted_solution
    """

    def mutate_split(self):
        solution_copy = np.array(self.solution.copy())
        values, cnt = np.unique(solution_copy, return_counts=True)
        val = np.random.choice(values, p= cnt/cnt.sum())
        idx = np.where(values == val)[0][0]
        if cnt[idx] > 1:
            i = cnt[idx]//2
            to_change = []
            for node in self.paths_dict[val][1:]:
                if solution_copy[node-1] == val:
                    to_change.append(node-1)
                if len(to_change) >= i:
                    dest = node
                    break
            solution_copy[to_change] = dest
        sol = Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, orig_paths_dict=self.orig_paths_dict, solution=solution_copy)
        #if sol.fitness() < self.fitness():
            #print("Better sol using split mutation:", sol.fitness())
        return sol

    def mutate_join(self):
        solution_copy = np.array(self.solution.copy())
        values, cnt = np.unique(solution_copy, return_counts=True)
        #dist_dict = {i:(np.sqrt((x-1/2)**2 + (y-1/2)**2)) for i, (x,y) in self.G.nodes(data='pos') if i in values}
        #weights = np.array([1/dist_dict[i] for i in values])
        prob = 1/cnt / (1/cnt).sum()
        #prob = weights / weights.sum()
        val = np.random.choice(values, p= prob)
        # from solution_copy, choose an index i whose value in the array != val but such that path_dict[i] contains val
        candidates = [u for i, u in enumerate(solution_copy) if u != val and val in self.paths_dict[u]]
        if len(candidates) > 0:
            # choose u as closest to origin among candidates
            #dist_dict = {i:(np.sqrt((x-1/2)**2 + (y-1/2)**2)) for i, (x,y) in self.G.nodes(data='pos') if i in candidates}
            #weights = np.array([1/dist_dict[i] for i in candidates])
            #p = weights / weights.sum()
            #u = np.random.choice(candidates, p=p)
            u = np.random.choice(candidates)
            solution_copy[solution_copy == val] = u
        sol = Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, orig_paths_dict=self.orig_paths_dict, solution=solution_copy)
        #if sol.fitness() < self.fitness():
            #print("Better sol using join mutation:", sol.fitness())
        return sol
    
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
        sol =  Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, orig_paths_dict=self.orig_paths_dict, solution=offspring)
        #if sol.fitness() < self.fitness() and sol.fitness() < p2.fitness():
        #    print("Better sol using crossover:", sol.fitness()) 
        return sol
    """
    def fitness(self):
        if self.fitness_value is not None:
            return self.fitness_value
        total_cost = 0.0
        W = nx.to_numpy_array(self.P.graph, nodelist=range(self.G.number_of_nodes()), weight='dist')
        for dest, path in self.paths_dict.items():
            orig_path = self.orig_paths_dict[dest]
            if dest == 0 or dest not in self.solution:
                continue
            for i in range(1, len(orig_path)):
                total_cost += W[orig_path[i-1]][orig_path[i]]
            nodes_to_grab = [i for i in path[1:] if self.solution[i-1] == dest]
            current_gold = 0.0
            if len(nodes_to_grab) == 1 and nodes_to_grab[0] == path[-1]:
                    return_path = orig_path
            else:
                return_path = path
            for i in range(len(return_path)-1, 0, -1):
                if return_path[i] in nodes_to_grab:
                    current_gold += self.gold_dict[return_path[i]]
                dist = W[return_path[i-1]][return_path[i]]
                #print(f"Going from {return_path[i]} to {return_path[i-1]} with distance {dist} and current gold {current_gold}")
                total_cost += dist + (self.P.alpha * dist * current_gold) ** self.P.beta
               
        self.fitness_value = total_cost
        return total_cost
    """
    
    def fitness(self):
        if self.fitness_value is not None:
            return self.fitness_value
        total_cost = 0.0
        W = nx.to_numpy_array(self.P.graph, nodelist=range(self.G.number_of_nodes()), weight='dist')

        for dest, path in self.paths_dict.items():
            orig_path = self.orig_paths_dict[dest]
            if dest == 0 or dest not in self.solution:
                continue
            for i in range(1, len(orig_path)):
                total_cost += W[orig_path[i-1]][orig_path[i]]
            nodes_to_grab = [i for i in path[1:] if self.solution[i-1] == dest]
            nearest_to_grab = min(nodes_to_grab, key=lambda x: path.index(x))
            current_gold = 0.0
            
            if len(nodes_to_grab) == 1 and nodes_to_grab[0] == dest:
                return_path = orig_path

            else:
                #print("dest", dest, "Nodes to grab:", nodes_to_grab,"path", path, "nearest to grab:", nearest_to_grab)
                return_path = self.orig_paths_dict[nearest_to_grab] + path[path.index(nearest_to_grab)+1:]
                #print("Return path:", return_path)

            for i in range(len(return_path)-1, 0, -1):
                if return_path[i] in nodes_to_grab:
                    current_gold += self.gold_dict[return_path[i]]
                dist = W[return_path[i-1]][return_path[i]]
                #print(f"Going from {return_path[i]} to {return_path[i-1]} with distance {dist} and current gold {current_gold}")
                total_cost += dist + (self.P.alpha * dist * current_gold) ** self.P.beta
               
        self.fitness_value = total_cost
        return total_cost
    
    
    def __str__(self):
        return str(self.solution)