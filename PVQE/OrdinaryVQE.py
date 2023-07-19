import sys

import pennylane as qml
from pennylane import numpy as np
import qiskit

import AnsatzGenerator

def train_vqe(H, ansatz_kwargs, init_theta=None, dev=None, stepsize=0.01, maxiter=300, tol=1e-6):
    # locals().update(ansatz_kwargs)
    # sys._getframe(1).f_locals.update(ansatz_kwargs)
    # print(set(locals()))
    # assert set(['ansatz_gen','num_qubits','num_layers']).issubset(set(locals()))
    
    num_qubits = ansatz_kwargs['num_qubits']
    num_layers = ansatz_kwargs['num_layers']
    ansatz_gen = ansatz_kwargs['ansatz_gen']

    print('num_qubits =', num_qubits)
    if not dev:
        dev = qml.device('default.qubit', wires=num_qubits+1)
    
    ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)
    
    @qml.qnode(dev, interface='autograd')
    def cost_fn(theta):
        ansatz_obj.get_ansatz(theta)
        return qml.expval(H)
    
    opt = qml.AdamOptimizer(stepsize=stepsize)
    if init_theta is None:
        theta = np.random.rand(ansatz_obj.num_parameters, requires_grad=True)
    else:
        theta = init_theta
        theta.requires_grad = True
        
    history = {"energy": [], "theta":[]}

    for n in range(maxiter):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        
        history['energy'].append(cost_fn(theta))
        history['theta'].append(theta)
        conv = np.abs(history['energy'][-1] - prev_energy)

        # if n % 10 == 0:
        #     print(f"Step = {n},  Energy = {history['energy'][-1]:.8f}")
        if conv  <= tol:
            break

    return history

if __name__ == '__main__':
    coeffs = [0.2, -0.543]
    obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
    H = qml.Hamiltonian(coeffs, obs)
    print(H.ops)
    ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':2, 'num_layers':3}
    record = train_vqe(H, ansatz_kwargs, maxiter=1000)
    print(record)




