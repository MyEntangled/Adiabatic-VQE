import pennylane as qml
import pennylane.numpy as np

from PVQE import AnsatzGenerator

class VQEApproximator:
    def __init__(self, num_qubits, num_layers, ansatz_gen, dev=None) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.ansatz_gen = ansatz_gen
        if not dev:
            self.dev = qml.device('default.qubit', wires=num_qubits)
        else:
            self.dev = dev

        self.ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)
        
    def update(self, H_prev, H_curr, theta_opt_prev):#, theta_init_curr):
        @qml.qnode(self.dev)
        def circuit(theta, H):
            self.ansatz_obj.get_ansatz(theta)
            return qml.expval(H)

        # grad_opt_prev = qml.jacobian(circuit)(theta_opt_prev, H_prev)
        # grad_opt_curr = grad_opt_prev ## Expectation
        # grad_init_curr = qml.jacobian(circuit)(theta_init_curr, H_curr)
        # hessian = qml.jacobian(qml.jacobian(circuit))(theta_init_curr, H_curr)

        # # hessian * (theta_opt_curr - theta_init_curr) = grad_opt_curr - grad_init_curr
        # h = np.linalg.solve(hessian, grad_opt_curr - grad_init_curr)
        # theta_opt_curr = h + theta_init_curr

        grad = qml.jacobian(circuit)(theta_opt_prev, H_curr)
        hessian = qml.jacobian(qml.jacobian(circuit))(theta_opt_prev, H_curr)
        h = np.linalg.solve(hessian, -grad)
        theta_init_curr


        return theta_opt_curr

class ParameterInitializer:
    def __init__(self, num_qubits, num_layers, ansatz_gen, dev=None) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.ansatz_gen = ansatz_gen
        if not dev:
            self.dev = qml.device('default.qubit', wires=num_qubits)
        else:
            self.dev = dev

        self.ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)

    def initialize(self, H_curr, theta_prev):
        @qml.qnode(self.dev)
        def circuit(theta, H):
            self.ansatz_obj.get_ansatz(theta)
            return qml.expval(H)

        grad = qml.jacobian(circuit)(theta_prev, H_curr)
        hessian = qml.jacobian(qml.jacobian(circuit))(theta_prev, H_curr)
        h = np.linalg.solve(hessian, -grad)
        theta_init = h + theta_prev
        return theta_init

class QAOAApproximator:
    def __init__(self, num_qubits, num_layers, ansatz_gen, H_mixer, dev=None) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.ansatz_gen = ansatz_gen
        if not dev:
            self.dev = qml.device('default.qubit', wires=num_qubits)
        else:
            self.dev = dev

        self.ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)
        self.H_mixer = H_mixer

    def update(self, H_prev, H_curr, theta_opt_prev, theta_init_curr):
        @qml.qnode(self.dev)
        def circuit(theta, H):
            gamma = theta[:self.ansatz_obj.num_layers]
            beta = theta[self.ansatz_obj.num_layers:]
            self.ansatz_obj.get_ansatz(gamma, beta, H, self.H_mixer)
            return qml.expval(H)

        grad_opt_prev = qml.jacobian(circuit)(theta_opt_prev, H_prev)
        grad_opt_curr = grad_opt_prev ## Expectation
        grad_init_curr = qml.jacobian(circuit)(theta_init_curr, H_curr)
        hessian = qml.jacobian(qml.jacobian(circuit))(theta_init_curr, H_curr)

        # hessian * (theta_opt_curr - theta_init_curr) = grad_opt_curr - grad_init_curr
        h = np.linalg.solve(hessian, grad_opt_curr - grad_init_curr)
        theta_opt_curr = h + theta_init_curr

        return theta_opt_curr