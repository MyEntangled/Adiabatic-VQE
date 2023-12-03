import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import copy
import Calculating_Expectation_Values as Expectation_Values
from Generating_Problems import MIS
import networkx as nx
from classical_solver import find_mis

class QIRO(Expectation_Values.ExpectationValues):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, problem_input, nc, strategy,no_correlation,temperature):
        super().__init__(problem=problem_input)
        # let us use the problem graph as the reference, and this current graph as the dynamic
        # object from which we will eliminate nodes:
        self.graph = copy.deepcopy(self.problem.graph)
        self.nc = nc
        self.assignment = []
        self.solution = []
        self.strategy = strategy
        self.no_correlation = no_correlation
        self.temperature = temperature

    
    def update_single(self, variable_index, exp_value_sign):
        """Updates Hamiltonian according to fixed single point correlation"""
        node = variable_index - 1
        fixing_list = []
        assignments = []
        # if the node is included in the IS we remove its neighbors
        if exp_value_sign == 1:
            ns = copy.deepcopy(self.graph.neighbors(node))
            for n in ns:
                self.graph.remove_node(n)
                fixing_list.append([n + 1])
                assignments.append(-1)
        
        # in any case we remove the node which was selected by correlations:
        self.graph.remove_node(node)
        fixing_list.append([variable_index])
        assignments.append(exp_value_sign)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        return fixing_list, assignments
    
    def update_correlation(self, variables, exp_value_sign):
        """Updates Hamiltonian according to fixed two point correlation -- RQAOA (for now)."""
        
        #     """This does the whole getting-of-coupled-vars mumbo-jumbo."""
        fixing_list = []
        assignments = []
        if exp_value_sign == 1:
            # if variables are correlated, then we set both to -1 
            # (as the independence constraint prohibits them from being +1 simultaneously). 
            for variable in variables:
                fixing_list.append([variable])
                assignments.append(-1)
                self.graph.remove_node(variable - 1)                
        else:
#             print("Entered into anticorrelated case:")
            # we remove the things we need to remove are the ones connected to both node, which are not both node.
            mutual_neighbors = set(self.graph.neighbors(variables[0] - 1)) & set(self.graph.neighbors(variables[1] - 1))
            fixing_list = [[n + 1] for n in mutual_neighbors]
            assignments = [-1] * len(fixing_list)
            for n in mutual_neighbors:
                self.graph.remove_node(n)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        return fixing_list, assignments
    

    def prune_graph(self):
        """Prunes the graph by removing all connected components that have less than nc nodes. The assignments are determined
        to be the maximum independent sets of the connected components. The self.graph is updated correspondingly."""

        # get connected components
        connected_components = copy.deepcopy(list(nx.connected_components(self.graph)))
        prune_assignments = {}
        for component in connected_components:
            if len(component) <= self.nc:
                subgraph = self.graph.subgraph(component)
                _, miss = find_mis(subgraph)
                prune_assignments.update({n: 1 if n in miss[0] else -1 for n in subgraph.nodes}) 

        # remove component from graph
        for node in prune_assignments.keys():
            self.graph.remove_node(node)

        self.problem = MIS(self.graph, self.problem.alpha)

        fixing_list = [[n + 1] for n in sorted(prune_assignments.keys())]
        assignments = [prune_assignments[n] for n in sorted(prune_assignments.keys())]

        return fixing_list, assignments
    
    def remove_duplication(self,exp_value_coeffs,exp_value_signs,exp_values):
        """ We remove duplication node in exp_value_coeffs. For instance, if we has exp_value_coeffs = [(1,2),2,3,4],
        We remove the node 2 to obtain exp_value_coeffs = [(1,2),3,4] 
        """
        unique_list = [] 
        filter_exp_value_coeffs = []
        filter_exp_value_signs = [] 
        filter_exp_values = [] 
        for index in range(len(exp_value_coeffs)):
            element = exp_value_coeffs[index] 
            if len(element) == 2: 
                node_large = element[0] 
                node_small = element[1] 
                if (node_large not in unique_list) and (node_small not in unique_list): 
                    unique_list.append(node_large) 
                    unique_list.append(node_small) 
                    filter_exp_value_coeffs.append(element) 
                    filter_exp_value_signs.append(exp_value_signs[index])
                    filter_exp_values.append(exp_values[index]) 
            else: 
                node = element[0] 
                if node not in unique_list: 
                    unique_list.append(node) 
                    filter_exp_value_coeffs.append(element) 
                    filter_exp_value_signs.append(exp_value_signs[index])
                    filter_exp_values.append(exp_values[index])                     
        return filter_exp_value_coeffs,filter_exp_value_signs, filter_exp_values 
    
    def remove_neighbor(self,exp_value_coeffs,exp_value_signs,exp_values):
        """ This function is used to remove node/(pair of node) which share neighbor
        such that we do not have confliction in the update rules"""
        def update_flag(element): 
            """ The auxilary function to determine if we add the element"""
            for existing_element in filter_exp_value_coeffs: 
                if len( neighbor_dict.get(tuple(existing_element)) & neighbor_dict.get(tuple(element)) ) > 0: 
                    return False
            return True 
        
        filter_exp_value_coeffs = []
        filter_exp_value_signs = [] 
        filter_exp_values = [] 
        
        # Create a dictionary which store neighbor for each element 
        neighbor_dict  = dict() 
        for element in exp_value_coeffs: 
            if len(element) == 2:
                # This -1 is the relic from above 
                node1 = element[0] - 1
                node2 = element[1] - 1
                # The tuple is due to python being crazy about list not hashable so we need this dumb work around 
                neighbor_dict[tuple(element)] = set( self.graph.neighbors(node1) ).union( set(self.graph.neighbors(node2)), set([node1,node2]))
            else: 
                # yeah i dont know why this is a thing but it is a mumbo jumbo from above 
                node = element[0] - 1
                neighbor_dict[tuple(element)] = set( self.graph.neighbors(node)).union(set([node])) 
        
        # We can always add the first element and add others conditioning on the first elements 
        filter_exp_value_coeffs.append(exp_value_coeffs[0])
        filter_exp_value_signs.append(exp_value_signs[0])
        filter_exp_values.append(exp_values[0])
        
        # Filtering 
        for index in range(1,len(exp_value_coeffs)):
            element = exp_value_coeffs[index] 
            if update_flag(element): 
                filter_exp_value_coeffs.append(exp_value_coeffs[index])
                filter_exp_value_signs.append(exp_value_signs[index])
                filter_exp_values.append(exp_values[index])
                
        return filter_exp_value_coeffs, filter_exp_value_signs, filter_exp_values            
        
        
        
        
        

    def execute(self, energy='best'):
        """Main QIRO function which produces the solution by applying the QIRO procedure."""
        self.opt_gamma = []
        self.opt_beta = []
        self.fixed_correlations = []
        step_nr = 0

        while self.graph.number_of_nodes() > 0:
            step_nr += 1
            print(f"Step: {step_nr}. Number of nodes: {self.graph.number_of_nodes()}.")
            # nx.draw(self.graph, with_labels = True)
            # plt.show()
            fixed_variables = []            
            
            exp_value_coeffs, exp_value_signs, exp_values, back_up_element = self.optimize(self.strategy,self.no_correlation,self.temperature)
            
#             print(f'Raw: {exp_value_coeffs}')
            exp_value_coeffs, exp_value_signs, exp_values = self.remove_duplication(exp_value_coeffs,
                                                                                    exp_value_signs, exp_values)
            exp_value_coeffs, exp_value_signs, exp_values = self.remove_neighbor(exp_value_coeffs,
                                                                                    exp_value_signs, exp_values)           
#             print(f'Filter: {exp_value_coeffs}')

            for index in range(len(exp_value_coeffs)): 
                exp_value_coeff = exp_value_coeffs[index]
                exp_value_sign = exp_value_signs[index]
                exp_value = exp_values[index]
                if len(exp_value_coeff) == 1: 
#                     print(f"single var {exp_value_coeff}. Sign: {exp_value_sign}")  
                    holder_fixed_variables, assignments = self.update_single(*exp_value_coeff,exp_value_sign)
                    fixed_variables += holder_fixed_variables 
                    for var, assignment in zip(holder_fixed_variables,assignments): 
                        self.fixed_correlations.append([var,int(assignment),exp_value])
                else:
#                     print(f"Correlation {exp_value_coeff}. Sign: {exp_value_sign}")
                    holder_fixed_variables, assignments = self.update_correlation(exp_value_coeff,exp_value_sign) 
                    fixed_variables += holder_fixed_variables 
                    for var, assignment in zip(holder_fixed_variables,assignments):
                        self.fixed_correlations.append([var,int(assignment),exp_value])

            # perform pruning.
            pruned_variables, pruned_assignments = self.prune_graph()
#             print(f"Pruned {len(pruned_variables)} variables.")
            for var, assignment in zip(pruned_variables, pruned_assignments):
                if var is None:
                    raise Exception("Variable to be eliminated is None. WTF?")
                self.fixed_correlations.append([var, assignment, None])
            fixed_variables += pruned_variables
            # Backup procedure in case no variable is fixed 
            if len(fixed_variables) == 0: 
                index = list(back_up_element[0])[0]
                backup_coeff = [self.problem.position_translater[index]]
                backup_sign  = np.sign(back_up_element[1]).astype(int) 
                backup_value = back_up_element[1]
#                 print(f"No fixed variables have been found, attempting with {backup_coeff}. Sign: {backup_sign}. Value: {backup_value}")
                holder_fixed_variables, assignments = self.update_single(*backup_coeff,backup_sign)
                fixed_variables += holder_fixed_variables 
                for var, assignment in zip(holder_fixed_variables,assignments): 
                    self.fixed_correlations.append([var,int(assignment),exp_value])
                        
        solution = [var[0] * assig for var, assig, _ in self.fixed_correlations]
        sorted_solution = sorted(solution, key=lambda x: abs(x))
        # print(f"Solution: {sorted_solution}")
        self.solution = np.array(sorted_solution).astype(int)

