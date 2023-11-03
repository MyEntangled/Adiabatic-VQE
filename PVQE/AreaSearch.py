import numpy as np
import pennylane as qml
from helper import *

class TriangleSearch:
    def __init__(self, num_qubits, pauli_words):
        self.num_qubits = num_qubits
        self.pauli_words = pauli_words

    def uniform_triangle(self, u, v):
        while True:
            s,t = np.random.rand(2)
            in_triangle = s + t <= 1
            p = s * u + t * v if in_triangle else (1 - s) * u + (1 - t) * v
            return p

    def meanfield_spectral_max(self, w0, w1, w2, radius):
        # w in conv(w0, w1, w2) and |w - w0| <= radius

        dir1 = w1 - w0
        len1 = np.linalg.norm(dir1)
        ndir1 = dir1 / len1

        dir2 = w2 - w0
        len2 = np.linalg.norm(dir2)
        ndir2 = dir2 / len2

        num_samples = 100
        gap_mf = []
        all_p = []
        for i in range(num_samples):
            p = w0 + self.uniform_triangle(radius * ndir1, radius * ndir2)
            all_p.append(p)
            H = qml.Hamiltonian(p, self.pauli_words)
            gap_mf.append(meanfield_spectral_gap(self.num_qubits, H))

        return all_p[np.argmax(gap_mf)]

        


