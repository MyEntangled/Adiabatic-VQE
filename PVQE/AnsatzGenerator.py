import pennylane as qml

class SimpleAnsatz:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.wires = range(num_qubits)
        self.num_parameters = 3*self.num_qubits*self.num_layers + 2*num_qubits

    def get_ansatz(self, theta):
        iter_theta = iter(theta)
        for l in range(self.num_layers):
            for i in self.wires:
                qml.RX(next(iter_theta), wires=i)
            for i in self.wires:
                qml.RX(next(iter_theta), wires=i)
            for i in self.wires:
                qml.CRZ(next(iter_theta), wires=[i,(i+1)%self.num_qubits])

        for i in self.wires:
            qml.RX(next(iter_theta), wires=i)
        for i in self.wires:
            qml.RY(next(iter_theta), wires=i)
        

