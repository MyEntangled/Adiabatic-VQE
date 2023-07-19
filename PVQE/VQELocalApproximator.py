import pennylane as qml
import pennylane.numpy as np

import AnsatzGenerator


def update_approximator(ansatz_kwargs, H_prev, H_curr, theta_opt_prev, theta_init_curr, dev=None):
    num_qubits = ansatz_kwargs['num_qubits']
    num_layers = ansatz_kwargs['num_layers']
    ansatz_gen = ansatz_kwargs['ansatz_gen']

    if not dev:
        dev = qml.device('default.qubit', wires=num_qubits+1)

    ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)

    @qml.qnode(dev)
    def circuit(theta, H):
        ansatz_obj.get_ansatz(theta)
        return qml.expval(H)

    grad_opt_prev = qml.jacobian(circuit)(theta_opt_prev, H_prev )
    grad_opt_curr = grad_opt_prev ## Expectation
    grad_init_curr = qml.jacobian(circuit)(theta_init_curr, H_curr)
    hessian = qml.jacobian(qml.jacobian(circuit))(theta_init_curr, H_curr)

    # hessian * (theta_opt_curr - theta_init_curr) = grad_opt_curr - grad_init_curr
    h = np.linalg.solve(hessian, grad_opt_curr - grad_init_curr)
    theta_opt_curr = h + theta_init_curr

    return theta_opt_curr


