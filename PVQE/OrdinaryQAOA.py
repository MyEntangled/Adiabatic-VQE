import sys

import pennylane as qml
from pennylane import numpy as np
import qiskit

import AnsatzGenerator

def train_qaoa(Hp, Hb, ansatz_kwargs, init_gamma=None, init_beta=None, dev=None, stepsize=0.01, maxiter=300, tol=1e-6, meas_terms = None):
    # if init_gamma is not None and init_beta is not None:
    #     print("init dimension", init_gamma.shape, init_beta.shape)
    num_qubits = ansatz_kwargs['num_qubits']
    num_layers = ansatz_kwargs['num_layers']
    if 'wires' in ansatz_kwargs:
        wires = ansatz_kwargs['wires']
    else:
        wires = list(range(num_qubits))


    #print('num_qubits =', num_qubits)
    if not dev:
        dev = qml.device('default.qubit', wires)

    ansatz_obj = getattr(AnsatzGenerator,"QAOAAnsatz")(num_qubits, num_layers)
    
    @qml.qnode(dev, interface='autograd', diff_method='best')
    def cost_fn(theta):
        gamma = theta[:ansatz_obj.num_layers]
        beta = theta[ansatz_obj.num_layers:]
        ansatz_obj.get_ansatz(gamma, beta, Hp, Hb, wires)
        if not measure_all:
            return qml.expval(Hp)
        else:
            #print("HELLO")
            return [qml.expval(term) for term in meas_terms]
    
    opt = qml.AdamOptimizer(stepsize=stepsize)
    if init_gamma is None:
        gamma = np.random.rand(ansatz_obj.num_layers, requires_grad=True)
    else:
        gamma = init_gamma
        gamma.requires_grad = True
    if init_beta is None:
        beta = np.random.rand(ansatz_obj.num_layers, requires_grad=True)
    else:
        beta = init_beta
        beta.requires_grad = True
        
    history = {"energy": [], "gamma": [], "beta": [], 'measurements': []}

    for n in range(maxiter):
        gamma.requires_grad = True
        beta.requires_grad = True
        measure_all = False
        theta, prev_energy = opt.step_and_cost(cost_fn, np.concatenate((gamma, beta)))
        gamma = theta[:ansatz_obj.num_layers]
        beta = theta[ansatz_obj.num_layers:]

        history['energy'].append(cost_fn(theta))
        history['gamma'].append(gamma)
        history['beta'].append(beta)

        if meas_terms is not None:
            gamma.requires_grad = False
            beta.requires_grad = False
            measure_all = True
            history['measurements'].append(cost_fn(theta))
            #print(history['measurements'][-1])

        conv = np.abs(history['energy'][-1] - prev_energy)
        #print(history['energy'][-1])
        if n % 20 == 0:
            print(f"Step = {n},  Energy = {history['energy'][-1]:.8f}")
        if conv  <= tol:
            break
    return history



