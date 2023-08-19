import pennylane as qml
from pennylane import qaoa

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
        
    
class QAOAAnsatz:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_parameters = 2*num_layers
    
    def get_ansatz(self, gamma, beta, Hp, Hb, wires=None):
        assert len(gamma) == len(beta)
        assert len(gamma) == self.num_layers
        if wires is None:
            wires = range(self.num_qubits)
        assert len(wires) == self.num_qubits

        for q in wires:
            qml.Hadamard(wires=q)

        for i in range(self.num_layers):
            #qml.evolve(Hp, coeff=gamma[i])
            #qaoa.cost_layer(gamma[i], Hp)
            qml.CommutingEvolution(Hp, gamma[i])

            #qml.evolve(Hb, coeff=beta[i])
            #qaoa.mixer_layer(beta[i], Hb)
            qml.CommutingEvolution(Hb, beta[i])







