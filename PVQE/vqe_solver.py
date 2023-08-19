import pennylane as qml
import pennylane.numpy as np
from scipy.integrate import solve_ivp, odeint

import AnsatzGenerator, LocalObservables, GravityODE, OrdinaryVQE, LocalApproximator
import helper

import itertools

## Set up
num_qubits = 6
num_layers = 3
full_basis = LocalObservables.get_k_local_basis(num_qubits, 3)
num_terms = len(full_basis) // 10

#coeffs = np.array([0.2, -0.543, 0.3])
coeffs = np.random.rand(num_terms)

norm = np.linalg.norm(coeffs)
coeffs = coeffs / norm

#obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliY(2),  qml.PauliX(1)]
terms_str = np.random.choice(full_basis, size=num_terms, replace=False)
obs = [qml.pauli.string_to_pauli_word(str(each)) for each in terms_str]

H = qml.Hamiltonian(coeffs, obs)
L_list, H_coeffs = LocalObservables.get_hamiltonian_terms(num_qubits, H)
L_terms = H.ops
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

## Prepare for first iteration
theta_approximator = LocalApproximator.VQEApproximator(num_qubits, start_ansatz_kwargs['num_layers'],
                                                       start_ansatz_kwargs['ansatz_gen'], dev=None)
num_iterations = 1000
training_record = {'dist-to-H':[], 'dist-to-prev': []}


prev_w = np.zeros(len(L_list), dtype=float)
for i,L_str in enumerate(L_list):
    if L_str in L_1q_list:
        prev_w[i] = H_coeffs[i]
    else:
        prev_w[i] = 0


## Iterates...
for it in range(num_iterations):
    print('Iteration: ', it+1)

    if it == 1:
        theta_approximator = LocalApproximator.VQEApproximator(num_qubits, ansatz_kwargs['num_layers'], 
                                                               ansatz_kwargs['ansatz_gen'], dev=None)

    print(prev_w)
    H_it = qml.Hamiltonian(prev_w, L_terms)

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


    # Set attribute for ODE Solver at the class level
    GravityODE.FirstOrderAttraction.converge_cond.terminal = True
    GravityODE.FirstOrderAttraction.converge_cond.direction = -1
    GravityODE.FirstOrderAttraction.deviation_bound.terminal = True
    GravityODE.FirstOrderAttraction.deviation_bound.direction = -1

    ## ODE solver
    ode_solver = GravityODE.FirstOrderAttraction(t_span=[0,10], H_coeffs=H_coeffs)
    w = ode_solver.solve_ivp(init_z=prev_w, nullspace=nullspace, prev_w=prev_w, time_step=0.05)

    #ode_solver = GravityODE.SecondOrderAttraction(t_span=[0,10], H_coeffs=H_coeffs)
    #w = ode_solver.solve_ivp(init_z=prev_w, nullspace=nullspace, prev_w=prev_w, time_step=0.05)
    
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
    