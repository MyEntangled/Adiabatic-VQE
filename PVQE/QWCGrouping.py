import pennylane as qml
import pennylane.numpy as np
import numpy as np
import networkx as nx
import warnings
import matplotlib.pyplot as plt
import LocalObservables
import random

def extend_group(pauli_term, other_terms, other_strings=None, wire_map=None):
    if other_strings is None:
        other_strings = [qml.pauli.pauli_word_to_string(term, wire_map) for term in other_terms]

    sampling_count = 0
    group = [pauli_term]

    while sampling_count < 1000 and len(other_terms) > 0:
        sampling_count += 1
        #print(len(other_terms))
        sample_id = random.choice(range(len(other_terms)))
        sample_term = other_terms[sample_id]
        sample_string = other_strings[sample_id]

        if qml.pauli.are_pauli_words_qwc([pauli_term, sample_term]):
            group.append(sample_term)
            del other_terms[sample_id]
            del other_strings[sample_id]
            sampling_count = 0
    return group, other_terms, other_strings


def grouping_pauli_terms(pauli_terms, pauli_strings, wire_map, random=False):
    if random == False:
        return qml.pauli.group_observables(pauli_terms, grouping_type='qwc', method='rlf')
    else:
        other_terms = pauli_terms.copy()
        other_strings = pauli_strings.copy()
        groups = []
        num_terms = 0

        while len(other_terms) > 0:
            first_term = other_terms[0]
            first_string = other_strings[0]

            other_terms = other_terms[1:]
            other_strings = other_strings[1:]
            group, other_terms, other_strings = extend_group(first_term, other_terms, other_strings)
            groups.append(group)
            num_terms += len(group)

        assert num_terms == len(pauli_terms)
        return groups



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
    num_qubits = 14
    num_layers = 3
    full_basis = LocalObservables.get_k_local_basis(num_qubits, 4)
    num_terms = len(full_basis) 
    wire_map = dict(zip(range(num_qubits), range(num_qubits)))

    #coeffs = np.array([0.2, -0.543, 0.3])
    coeffs = np.random.rand(num_terms)

    norm = np.linalg.norm(coeffs)
    coeffs = coeffs / norm
    #print(coeffs)
    #obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliY(2),  qml.PauliX(1)]
    pauli_strings = np.random.choice(full_basis, size=num_terms, replace=False)
    pauli_strings = list(pauli_strings)
    pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]
    #print('all terms:', pauli_strings)
    
    # group, weight = select_qwc_group(pauli_terms, terms_strings, coeffs)
    # print(group, weight)

    groups = grouping_pauli_terms(pauli_terms, pauli_strings, wire_map, random=True)
    groups_strings = [[qml.pauli.pauli_word_to_string(term, wire_map) for term in group] for group in groups]
    for group in groups:
        assert qml.pauli.are_pauli_words_qwc(group)
    print('num_terms', num_terms)
    print('num_terms_in_groups', sum([len(group) for group in groups_strings]))
    print(groups_strings)

    

