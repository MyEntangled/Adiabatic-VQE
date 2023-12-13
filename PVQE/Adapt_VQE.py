import pennylane as qml
from pennylane import qchem
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
#from joblib import Parallel, delayed


def train_adapt_vqe(molecule, observable, num_qubits, init_theta=None, dev=None, stepsize=0.01, maxiter=300, tol=1e-8):
    active_electrons = molecule.n_electrons
    singles, doubles = qchem.excitations(active_electrons, num_qubits)

    hf_state = qchem.hf_state(active_electrons, num_qubits)
    
    if isinstance(observable, qml.Hamiltonian):
        H = observable
    else:
        term_groups, coeff_groups = observable
        coeff_groups_flat = jnp.array(np.concatenate(coeff_groups))

    if not dev:
        dev = qml.device('lightning.qubit', wires=num_qubits)   
        
    ### Circuit to get all excitation operator gradient
    def circuit_1(params, excitations):
        qml.BasisState(hf_state, wires=range(num_qubits))

        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                qml.DoubleExcitation(params[i], wires=excitation)
            else:
                qml.SingleExcitation(params[i], wires=excitation)
        return qml.expval(H)
    
    cost_fn_1 = qml.QNode(circuit_1, dev, interface="jax")

    circuit_gradient = jax.grad(cost_fn_1, argnums=0)

    params = jnp.array([0.0] * len(doubles))
    grads = circuit_gradient(params, excitations=doubles)

    # Select double excitation operators
    doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]
    

    #opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params_doubles = jnp.array(np.zeros(len(doubles_select)))
    params_doubles.requires_grad = True
    
    opt = optax.adam(learning_rate=stepsize)
    opt_state_params_doubles = opt.init(params_doubles)
    
    #@jax.jit
    def step_circ_1(theta, opt_state):
        prev_energy, grad_circuit = jax.value_and_grad(cost_fn_1)(theta, excitations=doubles_select)
        updates, opt_state = opt.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta
    
    for n in range(100):
        params_doubles = step_circ_1(params_doubles, opt_state_params_doubles)
    
    ### Circuit to get single excitation operator gradient
    def circuit_2(params, excitations, gates_select, params_select):
        qml.BasisState(hf_state, wires=range(num_qubits))

        for i, gate in enumerate(gates_select):
            if len(gate) == 4:
                qml.DoubleExcitation(params_select[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(params_select[i], wires=gate)

        for i, gate in enumerate(excitations):
            if len(gate) == 4:
                qml.DoubleExcitation(params[i], wires=gate)
            elif len(gate) == 2:
                qml.SingleExcitation(params[i], wires=gate)
        return qml.expval(H)
    
    # Compute the gradients for single excitation gates
    cost_fn_2 = qml.QNode(circuit_2, dev, interface="jax")
    circuit_gradient = jax.grad(cost_fn_2, argnums=0)
    params = jnp.array([0.0] * len(singles))

    grads = circuit_gradient(
        params,
        excitations=singles,
        gates_select=doubles_select,
        params_select=params_doubles
    )
    
    singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > 1.0e-5]
    
    cost_fn_1 = qml.QNode(circuit_1, dev, interface="jax")
    params = np.zeros(len(doubles_select + singles_select))
    gates_select = doubles_select + singles_select

    if init_theta is None:
        theta = jnp.array(np.random.uniform(low=0.0, high=2*np.pi, size=len(doubles_select + singles_select)))
        theta.requires_grad = True
    else:
        theta = jnp.array(params)
        theta.requires_grad = True
        
    opt_state = opt.init(theta)
    history = {"energy": [], "theta":[]}
    
    #@jax.jit
    def step_circ_2(theta, opt_state):
        prev_energy, grad_circuit = jax.value_and_grad(cost_fn_1)(theta, excitations=gates_select)
        updates, opt_state = opt.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta, prev_energy    
    
    for n in range(300):
        theta, energy = step_circ_2(theta, opt_state)
        if n%10 == 0:
            print(f"Iter {n} and Energy:{energy}")
        history['energy'].append(energy)
        history['theta'].append(theta)    
    return history