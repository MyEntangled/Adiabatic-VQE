import pennylane as qml
import pennylane.numpy as np


def true_ground_state_energy(H):
    eigvals, _ = np.linalg.eigh(qml.matrix(H))
    ref_value = eigvals[0]
    return eigvals[0], eigvals[1]