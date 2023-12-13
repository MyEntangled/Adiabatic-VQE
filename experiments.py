import numpy as np
from pennylane import qchem
import pennylane as qml
import PVQE 
import datasets
from PVQE import Solver, OrdinaryVQE
import pickle

def physical_systems(molecule:str, bond_lengths:list[int], num_layers:int, dir='results'):
    assert molecule in ['H2', 'LiH', 'HF']
    bond_lengths_dict = {'H2': [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.742, 0.78, 0.82, 0.86, 0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14, 1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46, 1.5, 1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86, 1.9, 1.94, 1.98, 2.02, 2.06, 2.1],
                         'LiH': [0.9, 0.93, 0.96, 0.99, 1.02, 1.05, 1.08, 1.11, 1.14, 1.17, 1.2, 1.23, 1.26, 1.29, 1.32, 1.35, 1.38, 1.41, 1.44, 1.47, 1.5, 1.53, 1.56, 1.57, 1.59, 1.62, 1.65, 1.68, 1.71, 1.74, 1.77, 1.8, 1.83, 1.86, 1.89, 1.92, 1.95, 1.98, 2.01, 2.04, 2.07, 2.1],
                         'HF': [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86, 0.9, 0.917, 0.94, 0.98, 1.02, 1.06, 1.1, 1.14, 1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.46, 1.5, 1.54, 1.58, 1.62, 1.66, 1.7, 1.74, 1.78, 1.82, 1.86, 1.9, 1.94, 1.98, 2.02, 2.06, 2.1]
}
    eqbm_bond_length = {'H2': 0.742, 'LiH': 1.57, 'HF': 0.917}
    
    if bond_lengths is not None:
        assert set(bond_lengths).issubset(bond_lengths_dict[molecule]), print(f'Input bond lengths should be in the following set {bond_lengths_dict[molecule]}')
        selected_bond_lengths = bond_lengths
    else:
        selected_bond_lengths = bond_lengths_dict[molecule][::3]
        if eqbm_bond_length[molecule] not in eqbm_bond_length:
            selected_bond_lengths = np.sort(selected_bond_lengths + [eqbm_bond_length[molecule]])

    for bond_length in selected_bond_lengths:
        data = qml.data.load("qchem", molname=molecule, basis="STO-3G", attributes=['hamiltonian','vqe_energy', 'fci_energy'], bondlength=bond_length)[0]
        H = data.hamiltonian
        num_qubits = len(H.wires)

        pauli_terms = H.ops
        coeffs = H.coeffs
        
        wire_map = dict(zip(range(num_qubits), range(num_qubits)))
        pauli_strings = [qml.pauli.pauli_word_to_string(term,wire_map) for term in pauli_terms]

        start_ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':0}
        ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':num_layers}
        solver = Solver.VQESolver(num_qubits, pauli_strings, coeffs, ansatz_kwargs, start_ansatz_kwargs)
        record = solver.solve(50)

        comps = [molecule, str(bond_length), 'VQE', str(num_layers)]
        filename = dir + '/' + '_'.join(comps) + '.p'

        pickle.dump(record, open( filename, "wb" ))


#physical_systems(molecule='HF', bond_lengths=None, num_layers=3)

data = qml.data.load("qchem", molname='He2', basis="6-31G", attributes=['molecule','hamiltonian','vqe_energy', 'fci_energy'], bondlength=0.65)[0]
H = data.hamiltonian
num_qubits = len(H.wires)
OrdinaryVQE.train_adaptive_vqe(H, data.molecule, num_qubits, grad_tol=1e-4)

    