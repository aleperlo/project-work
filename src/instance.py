import networkx as nx
import numpy as np
from collections import defaultdict

class Solution:
    """
    Class representing a solution for the problem instance.
    Attributes:
        P: Problem instance
        G: Simplified graph associated with the problem
        paths_dict: Dictionary of shortest paths from source to all nodes
        gold_dict: Dictionary of gold values for each node
        orig_paths_dict: Original paths dictionary from the problem graph
        admissible_values: Dictionary mapping each node to its admissible destination hubs
        admissible_mutations: Dictionary of nodes that can be mutated
        solution: Current solution representation as a list
        fitness_value: Cached fitness value of the solution
    Methods:
        random_solution: Generates a random valid solution
        format_solution: Formats the solution for output
        mutate_split: Mutation operation that splits a hub
        mutate_join: Mutation operation that joins hubs
        crossover: Crossover operation between two solutions
        fitness: Computes the fitness value of the solution
    """
    def __init__(self, P, G=None, paths_dict=None, gold_dict=None, orig_paths_dict=None, solution=None):
        """
        Initializes the Solution object.
        Args:
            P: Problem instance
            G: Simplified graph (optional)
            paths_dict: Precomputed paths dictionary (optional)
            gold_dict: Precomputed gold dictionary (optional)
            orig_paths_dict: Precomputed original paths dictionary (optional)
            solution: Initial solution (optional)
        """
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
        # Compute admissible values for each entry in the solution
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
        """
        Generates a random valid solution.
        Returns:
            A list representing the random solution.
        """
        solution = []
        for i in range(1, len(self.paths_dict)):
            # Choose a random admissible value for node i
            c = np.random.choice(self.admissible_values[i])
            solution.append(c.item())
        values = set(solution)
        # Repair strategy: ensure each hub points to itself
        for v in values:
            if v != solution[v-1]:
                solution[v-1] = v
        return solution

    def format_solution(self):
        """
        Formats the solution for output.
        Returns:
            A list of tuples representing the formatted solution.
        """
        formatted_solution = []
        for dest, path in self.paths_dict.items():
            if dest == 0:
                continue
            # Get the shortest path to the destination in the original graph
            orig_path = self.orig_paths_dict[dest]
            # Only process if the destination is an hub node
            if dest in self.solution:
                # Identify nodes to grab gold from when going to node 'dest'
                nodes_to_grab = [i for i in path[1:] if self.solution[i-1] == dest]
                # Find the nearest node to grab gold from
                nearest_to_grab = min(nodes_to_grab, key=lambda x: path.index(x))
                # Add the path to the destination, no gold collected on the way
                for node in orig_path[1:-1]:
                    formatted_solution.append((node, 0.0))
                # If only the destination node is to be grabbed, return using the original path
                if len(nodes_to_grab) == 1 and nodes_to_grab[0] == path[-1]:
                    return_path = orig_path
                # Otherwise, return using the path from nearest_to_grab + remaining path to dest on the simplified graph
                else:
                    return_path = self.orig_paths_dict[nearest_to_grab] + path[path.index(nearest_to_grab)+1:]
                # Add the return path, collecting gold as specified
                for node in return_path[len(return_path)-1::-1]:
                    gold = self.gold_dict[node] if node in nodes_to_grab else 0.0
                    formatted_solution.append((node, gold))
        return formatted_solution
    
    def mutate_split(self):
        """
        Performs a split mutation on the solution.
        Returns:
            A new Solution object after mutation.
        """
        solution_copy = np.array(self.solution.copy())
        # Select a value to split based on frequency
        values, cnt = np.unique(solution_copy, return_counts=True)
        val = np.random.choice(values, p= cnt/cnt.sum())
        idx = np.where(values == val)[0][0]
        # Perform split if count > 1
        if cnt[idx] > 1:
            # Split approximately half of the nodes to a new destination
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
        return sol

    def mutate_join(self):
        """
        Performs a join mutation on the solution.
        Returns:
            A new Solution object after mutation.
        """
        # Select a value to join based on inverse frequency
        solution_copy = np.array(self.solution.copy())
        values, cnt = np.unique(solution_copy, return_counts=True)
        prob = 1/cnt / (1/cnt).sum()
        val = np.random.choice(values, p= prob)
        # Find candidates to join with
        candidates = [u for _, u in enumerate(solution_copy) if u != val and val in self.paths_dict[u]]
        max_tries = 100
        tries = 0
        # Retry if no candidates found
        while len(candidates) <= 0 and tries < max_tries:
            val = np.random.choice(values)
            candidates = [u for _, u in enumerate(solution_copy) if u != val and val in self.paths_dict[u]]
            tries += 1
        # Perform join if candidates found
        if len(candidates) > 0:
            u = np.random.choice(candidates)
            solution_copy[solution_copy == val] = u
        sol = Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, orig_paths_dict=self.orig_paths_dict, solution=solution_copy)
        return sol
    
    def crossover(self, p2):
        """
        Performs crossover between this solution and another solution.
        Args:
            p2: Another Solution object to crossover with.
        Returns:
            A new Solution object after crossover.
        """
        offspring = np.zeros(len(self.solution), dtype=int)
        
        # Get unique destinations (hubs) from both parents
        v1 = set(np.unique(self.solution))
        v2 = set(np.unique(p2.solution))
        
        while np.any(offspring == 0):
            if not v1 and not v2:
                # Get indices of all unassigned (orphaned) nodes
                zeros_idx = np.where(offspring == 0)[0]
                for idx in zeros_idx:
                    # Create a self-loop for orphaned nodes
                    offspring[idx] = idx + 1
                break

            # --- Try Parent 1 ---
            if v1:
                val = np.random.choice(list(v1))
                v1.remove(val)
                hub_idx = val - 1  # Convert 1-based value to 0-based index
                
                # CONSISTENCY CHECK: 
                # Can 'val' be a hub? Only if its own slot is 0 or already 'val'.
                if offspring[hub_idx] == 0 or offspring[hub_idx] == val:
                    # Assign 'val' to all empty spots where Parent 1 had 'val'
                    mask = (self.solution == val) & (offspring == 0)
                    offspring[mask] = val
                    
                    # FORCE HUB: Ensure the destination itself points to itself
                    offspring[hub_idx] = val

            # --- Try Parent 2 ---
            if v2:
                val = np.random.choice(list(v2))
                v2.remove(val)
                hub_idx = val - 1
                
                # CONSISTENCY CHECK
                if offspring[hub_idx] == 0 or offspring[hub_idx] == val:
                    mask = (p2.solution == val) & (offspring == 0)
                    offspring[mask] = val
                    offspring[hub_idx] = val

        sol = Solution(P=self.P, paths_dict=self.paths_dict, gold_dict=self.gold_dict, orig_paths_dict=self.orig_paths_dict, solution=offspring)
        return sol
   
    def fitness(self):
        """
        Computes the fitness value of the solution.
        Returns:
            The fitness value as a float.
        """
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
                return_path = self.orig_paths_dict[nearest_to_grab] + path[path.index(nearest_to_grab)+1:]

            for i in range(len(return_path)-1, 0, -1):
                if return_path[i] in nodes_to_grab:
                    current_gold += self.gold_dict[return_path[i]]
                dist = W[return_path[i-1]][return_path[i]]
                total_cost += dist + (self.P.alpha * dist * current_gold) ** self.P.beta
               
        self.fitness_value = total_cost
        return total_cost
    
    
    def __str__(self):
        """String representation of the solution."""
        return str(self.solution)