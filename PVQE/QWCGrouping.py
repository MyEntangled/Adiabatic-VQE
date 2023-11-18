import pennylane as qml
import pennylane.numpy as np
import numpy.random as random
import networkx as nx
import warnings
import matplotlib.pyplot as plt
import LocalObservables

def get_qwc_groups(pauli_terms, pauli_strings):
    G = nx.Graph()

    with warnings.catch_warnings():
        # Muting irrelevant warnings
        warnings.filterwarnings(
            "ignore",
            message="The behaviour of operator ",
            category=UserWarning,
        )

        # add Pauli strings to the graph
        G.add_nodes_from(pauli_strings)

        # add edges for qwc operators
        num_terms = len(pauli_strings)
        for i in range(num_terms):
            for j in range(i+1,num_terms):
                if qml.pauli.are_pauli_words_qwc([pauli_terms[i], pauli_terms[j]]):
                    G.add_edge(pauli_strings[i],pauli_strings[j])

        C = nx.complement(G)
    
        groups_dict = nx.coloring.greedy_color(C, strategy="largest_first")

    num_groups = len(set(groups_dict.values()))
    groups = []
    for i in range(num_groups):
        groups.append([pauli_str for pauli_str, group_id in groups_dict.items() if group_id == i])

    return groups, groups_dict

def select_qwc_group(pauli_terms, pauli_strings, coeffs, criterion='max-l2-norm'):
    assert len(pauli_terms) == len(pauli_strings)
    assert len(pauli_terms) == len(coeffs)

    groups, groups_dict = get_qwc_groups(pauli_terms, pauli_strings)
    num_groups = len(groups)

    if criterion == 'max-l2-norm':
        weights = np.zeros(num_groups)

        for m in range(len(pauli_terms)):
            group_id = groups_dict[pauli_strings[m]]
            weights[group_id] += coeffs[m]**2
        
        max_id = np.argmax(weights)
        return groups[max_id], weights[max_id]

    
if __name__ == '__main__':
    ## Set up
    num_qubits = 6
    num_layers = 3
    full_basis = LocalObservables.get_k_local_basis(num_qubits, 3)
    num_terms = len(full_basis) // 10

    #coeffs = np.array([0.2, -0.543, 0.3])
    coeffs = np.random.rand(num_terms)

    norm = np.linalg.norm(coeffs)
    coeffs = coeffs / norm
    print(coeffs)
    #obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliY(2),  qml.PauliX(1)]
    terms_str = random.choice(full_basis, size=num_terms, replace=False)
    obs = [qml.pauli.string_to_pauli_word(str(each)) for each in terms_str]
    print('all terms:', terms_str)
    group, weight = select_qwc_group(obs, terms_str, coeffs)
    print(group, weight)

    

