import AnsatzGenerator, Solver, LocalObservables
import numpy as np
import scipy
import pennylane as qml
from pennylane import qchem
import jax
import jax.numpy as jnp
from joblib import Parallel, delayed
import helper
import matplotlib.pyplot as plt

def path_function(time_points, coeffs_list):
    time_duration = len(coeffs_list) - 1
    path_coeffs = []
    for s in time_points:
        if s == time_duration:
            path_coeffs.append(coeffs_list[-1])
        else:
            assert np.floor(s) >= 0 and s <= time_duration
            path_coeffs.append((1.-s%1)*coeffs_list[int(np.floor(s))] + (s%1)*coeffs_list[int(np.floor(s))+1])
    return path_coeffs

    
def discretize_evolution(coeffs_list, pauli_terms, time_step):
    time_duration = len(coeffs_list)-1
    time_points = np.arange(0, time_duration + time_step, time_step)
    path_coeffs = path_function(time_points, coeffs_list)
    for t,s in enumerate(time_points):
        if s - int(s) == 0:
            print(f'time = {s}:', path_coeffs[t])

    path_H = [qml.Hamiltonian(coeffs, pauli_terms) for coeffs in path_coeffs]
    return path_H

def simulate_discrete_adiabatic_process(ansatz_kwargs, theta, time_step, path_H, H_prob, dev=None):
    num_qubits = ansatz_kwargs['num_qubits']
    num_layers = ansatz_kwargs['num_layers']
    ansatz_gen = ansatz_kwargs['ansatz_gen']

    if not dev:
        dev = qml.device('default.qubit', wires=num_qubits+1)
    
    ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)
    
    @qml.qnode(dev, interface='jax')
    def circuit(init_state, H, time_step):
        qml.QubitStateVector(init_state, wires=0)
        qml.evolve(H, time_step)
        return qml.state(), qml.expval(H)

    @qml.qnode(dev)
    def init(theta):
        ansatz_obj.get_ansatz(theta)
        return qml.state()
    
    energy_record = []
    init_state = init(theta)

    for H in path_H:
        init_state, energy = circuit(init_state, H, time_step)
        energy_record.append(energy)

    return energy_record


# np.random.seed(10)
# num_qubits = 3
# num_layers = 3
# full_basis = LocalObservables.get_k_local_basis(num_qubits, 2)
# num_terms = len(full_basis) 

# coeffs = np.random.rand(num_terms)
# coeffs = coeffs / np.linalg.norm(coeffs)
# coeffs = np.array([1,2,3,4])
# coeffs = coeffs / np.linalg.norm(coeffs)

# pauli_strings = np.random.choice(full_basis, size=num_terms, replace=False)
# pauli_strings = ['XII', 'XIZ', 'YIX', 'ZZY']
# print(pauli_strings)
# pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]
# H = qml.Hamiltonian(coeffs, pauli_terms)

symbols = ["H", "H"]
R = 1.2
coordinates = np.array([[0, 0, 0], [0, 0, R/0.529]])
H, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates)

num_layers = 4
coeffs = H.coeffs
coeffs = coeffs / np.linalg.norm(coeffs)
pauli_terms = H.ops
wire_map = dict(zip(range(num_qubits), range(num_qubits)))
pauli_strings = [qml.pauli.pauli_word_to_string(term,wire_map) for term in pauli_terms]
    
start_ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':0}
ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':num_layers}
solver = Solver.VQESolver(num_qubits, pauli_strings, coeffs, ansatz_kwargs, start_ansatz_kwargs)
record = solver.solve(50)
for t in range(len(record['H_list'])):
    print(f't = {t}', record['H_list'][t].coeffs)

time_step = 0.001
coeffs_list = [each.coeffs for each in record['H_list']]
path_H = discretize_evolution(coeffs_list, pauli_terms, time_step)
true_energies = [helper.true_ground_state_energy(H) for H in path_H]
energy_record = simulate_discrete_adiabatic_process(ansatz_kwargs, record['ground_theta'][0], time_step, path_H, H_prob=H)
diff = (np.array(energy_record) - np.array(true_energies))/np.array(true_energies)
print(diff)
print(len(diff))
time_points = np.arange(0, len(coeffs_list)-1 + time_step, time_step)
plt.plot(time_points, diff)
plt.show()
