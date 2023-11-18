import pennylane as qml
import pennylane.numpy as np
import numpy.random as random
from scipy.integrate import solve_ivp, odeint

import AnsatzGenerator, LocalObservables, GravityODE, OrdinaryVQE, PVQE.InitialParameters as InitialParameters, PVQE.MaxMFGapSearch as MaxMFGapSearch
import helper

import itertools

## Set up
num_qubits = 6
num_layers = 3
full_basis = LocalObservables.get_k_local_basis(num_qubits, 3)
num_terms = len(full_basis) // 10

coeffs = np.random.rand(num_terms)
norm = np.linalg.norm(coeffs)
coeffs = coeffs / norm

pauli_strings = random.choice(full_basis, size=num_terms, replace=False)
print(pauli_strings)
pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]
print(pauli_terms)

H = qml.Hamiltonian(coeffs, pauli_terms)
L_list, H_coeffs = LocalObservables.get_hamiltonian_terms(num_qubits, H)
# L_terms = H.ops
print("Pauli terms: ", L_list)
print("Coeffs:", H_coeffs)
print("Ground state energy = ", helper.true_ground_state_energy(H))

local_1q_basis = LocalObservables.get_k_local_basis(num_qubits, 1)
L_1q_list = set(local_1q_basis).intersection(set(L_list))

start_ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':0}
ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':num_layers}

exp_terms_str, exp_coeffs = LocalObservables.get_hamiltonian_terms(num_qubits, H)
cov_terms_str, cov_phases, cov_coeffs = LocalObservables.get_cov_terms(num_qubits, H)

meas_str = set(itertools.chain.from_iterable(cov_terms_str)).union(exp_terms_str)
meas_str = list(meas_str)

num_parameters = AnsatzGenerator.SimpleAnsatz(num_qubits, num_layers).num_parameters

## Prepare the first iteration
theta_approximator = InitialParameters.VQEApproximator(num_qubits, start_ansatz_kwargs['num_layers'],
                                                       start_ansatz_kwargs['ansatz_gen'], dev=None)
num_iterations = 1000
training_record = {'dist-to-H':[], 'dist-to-prev': []}


prev_w = np.zeros(len(L_list), dtype=float)
prev_w[0] = 1.

for i,L_str in enumerate(L_list):
    if L_str in L_1q_list:
        prev_w[i] = H_coeffs[i]
    else:
        prev_w[i] = 0


## Iterates...
#ode_solver = GravityODE.FirstOrderAttraction(t_span=[0,10], H_coeffs=H_coeffs)
#ode_solver = GravityODE.SecondOrderAttraction(t_span=[0,10], H_coeffs=H_coeffs)
area_searcher = MaxMFGapSearch.TriangleSearch(num_qubits, pauli_terms)

for it in range(num_iterations):
    print('Iteration: ', it+1)

    if it == 1:
        theta_approximator = InitialParameters.VQEApproximator(num_qubits, ansatz_kwargs['num_layers'], 
                                                               ansatz_kwargs['ansatz_gen'], dev=None)

    print(prev_w)
    H_it = qml.Hamiltonian(prev_w, pauli_terms)

    if it == 0:
        #res = VQE_numerical_solver(Hp_op, start_ansatz, num_ortho_init_states=1)
        history = OrdinaryVQE.train_vqe(H_it, start_ansatz_kwargs, stepsize=0.1)
        ground_energy = history['energy'][-1]
        start_ground_theta = history['theta'][-1]
        ground_theta = np.concatenate((start_ground_theta, np.zeros(num_parameters - len(start_ground_theta))))

    else:
        # approx_ground_theta = LocalApproximator.update_approximator(ansatz_kwargs, H_prev, H_it, theta_opt_prev=ground_theta, theta_init_curr=ground_theta)
        approx_ground_theta = theta_approximator.update(H_prev, H_it, theta_opt_prev=ground_theta, theta_init_curr=ground_theta)

        history = OrdinaryVQE.train_vqe(H_it, ansatz_kwargs, ground_theta)
        approx_history = OrdinaryVQE.train_vqe(H_it, ansatz_kwargs, approx_ground_theta)
        energy = history['energy'][-1]
        approx_energy = approx_history['energy'][-1]

        if energy < approx_energy:
            print("Ground theta WINS")
            print("Difference", approx_energy - energy)
            ground_energy = energy
            ground_theta = history['theta'][-1]
        else:
            print("Approx theta WINS")
            print("Difference", energy - approx_energy)
            ground_energy = approx_energy
            ground_theta = approx_history['theta'][-1]           

        # ground_energy = history['energy'][-1]
        # ground_theta = history['theta'][-1]
    print('True Ground Energy: ', helper.true_ground_state_energy(H_it))
    print('Est Ground Energy: ', ground_energy)
    
    meas_dict = LocalObservables.get_meas_outcomes(meas_str, ansatz_kwargs, ground_theta)
    M = LocalObservables.compute_correlation_matrix(meas_dict, exp_terms_str, cov_terms_str, cov_phases, is_weighted=False)
    
    ## Choose the firing direction == lowest eigenvector
    eigvals, eigvecs = np.linalg.eigh(M)
    nullspace = eigvecs[:,0]
    print('Nullspace: ', np.round(nullspace[:2], 3))


    # # Set attribute for ODE Solver at the class level
    # GravityODE.FirstOrderAttraction.converge_cond.terminal = True
    # GravityODE.FirstOrderAttraction.converge_cond.direction = -1
    # GravityODE.FirstOrderAttraction.deviation_bound.terminal = True
    # GravityODE.FirstOrderAttraction.deviation_bound.direction = -1
    ## ODE solver
    #w = ode_solver.solve_ivp(init_z=prev_w, nullspace=nullspace, prev_w=prev_w, time_step=0.05)

    ## Area Search
    w = area_searcher.meanfield_spectral_max(w0 = prev_w, w1 = nullspace, w2 = H_coeffs, radius = min(0.1, np.linalg.norm(prev_w - H_coeffs)))
    
    null_deviation = np.linalg.norm(M @ w)
    diff_to_H = np.linalg.norm(w - H_coeffs)
    diff_to_prev = np.linalg.norm(w - prev_w)

    ## Update record
    training_record['dist-to-H'].append(diff_to_H)
    training_record['dist-to-prev'].append(diff_to_prev)
    
    print('Prev w: ', prev_w[:3])
    print('Current w:', w[:3])
    print('Null-space deviation:', null_deviation)
    print('Difference to H: ', diff_to_H)
    print('Difference to prev: ', diff_to_prev)
    
    prev_w = w
    H_prev = H_it

    print('----')
    