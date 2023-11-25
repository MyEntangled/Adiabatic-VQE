import sys
import pennylane as qml
import numpy as np
#from pennylane import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import optax


import AnsatzGenerator

def train_vqe(observable, ansatz_kwargs, init_theta=None, dev=None, stepsize=0.01, maxiter=300, tol=1e-6):
    num_qubits = ansatz_kwargs['num_qubits']
    num_layers = ansatz_kwargs['num_layers']
    ansatz_gen = ansatz_kwargs['ansatz_gen']

    if isinstance(observable, qml.Hamiltonian):
        H = observable
    else:
        term_groups, coeff_groups = observable
        coeff_groups_flat = jnp.array(np.concatenate(coeff_groups))

    if not dev:
        dev = qml.device('default.qubit', wires=num_qubits+1)
    
    ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)
    
    @jax.jit
    @qml.qnode(dev, interface='jax')
    def cost_fn_H(theta):
        ansatz_obj.get_ansatz(theta)
        return qml.expval(H)
    
    @jax.jit
    @qml.qnode(dev, interface='jax')
    def cost_fn_group(theta, group):
        #print(group)
        ansatz_obj.get_ansatz(theta)
        return [qml.expval(o) for o in group]
    
    def cost_fn_groupings(theta):
        cost = 0
        # for i, group in enumerate(term_groups):
        #     group_outcomes = cost_fn_group(theta, group)
        #     cost += np.inner(coeff_groups[i],group_outcomes)
        group_outcomes = [jnp.array(cost_fn_group(theta, group)) for group in term_groups]
        outcomes = jnp.concatenate(group_outcomes)
        #print(outcomes)
        return jnp.inner(outcomes, coeff_groups_flat)
    
    if isinstance(observable, qml.Hamiltonian):
        cost_fn = cost_fn_H
    else:
        cost_fn = cost_fn_groupings

    if init_theta is None:
        theta = jnp.array(np.random.rand(ansatz_obj.num_parameters))
        theta.requires_grad = True
    else:
        theta = jnp.array(init_theta)
        theta.requires_grad = True
        
    history = {"energy": [], "theta":[]}

    #opt = qml.AdamOptimizer(stepsize=stepsize)
    opt = optax.adam(learning_rate=stepsize)
    opt_state = opt.init(theta)

    for n in range(maxiter):
        #theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        prev_energy, grad_circuit = jax.value_and_grad(cost_fn)(theta)
        updates, opt_state = opt.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)
        
        history['energy'].append(cost_fn(theta))
        history['theta'].append(theta)
        conv = np.abs(history['energy'][-1] - prev_energy)

        # if n % 10 == 0:
        #     print(f"Step = {n},  Energy = {history['energy'][-1]:.8f}")
        if conv  <= tol:
            break

    return history

def get_meas_outcomes(term_groups, string_groups, ansatz_kwargs, theta, dev=None):    
    theta = jnp.array(theta)    
    num_qubits = ansatz_kwargs['num_qubits']
    num_layers = ansatz_kwargs['num_layers']
    ansatz_gen = ansatz_kwargs['ansatz_gen']
    
    if not dev:
        dev = qml.device('default.qubit', wires=num_qubits)

    ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)

    @jax.jit
    @qml.qnode(dev, interface='jax')
    def circuit(theta, group=None):
        ansatz_obj.get_ansatz(theta)
        return [qml.expval(o) for o in group]
    
    #meas_outcomes = [circuit(theta, group) for group in term_groups]
    meas_dict = {}
    for id, group in enumerate(term_groups):
        outcomes = circuit(theta, group)
        meas_dict.update(zip(string_groups[id], outcomes))

    #meas_dict = dict(zip(meas_strings, meas_outcomes))
    return meas_dict

if __name__ == '__main__':
    coeffs = [0.2, -0.543]
    obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
    H = qml.Hamiltonian(coeffs, obs)
    print(H.ops)
    ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':2, 'num_layers':3}
    record = train_vqe(H, ansatz_kwargs, maxiter=1000)
    #print(record)




