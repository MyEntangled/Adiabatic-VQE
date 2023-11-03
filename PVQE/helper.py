import pennylane as qml
import pennylane.numpy as np


def true_ground_state_energy(H):
    eigvals, _ = np.linalg.eigh(qml.matrix(H))
    ref_value = eigvals[0]
    return eigvals[0], eigvals[1]

def meanfield_reduction(num_qubits, pauli_word, sym_mean, wire_map=None):
    if wire_map is None:
        wire_map = dict(zip(range(num_qubits), range(num_qubits)))
        
    order = pauli_word.num_wires
    
    pauli = qml.pauli.pauli_word_to_string(pauli_word, wire_map)
    
    gate_idx = {'X':0, 'Y':1, 'Z':2}
    
    expanded = []
    expanded_coeffs = []
    
    non_identity_idx = pauli_word.wires.labels
    
    for i in non_identity_idx:
        new_pauli = 'I'*i + pauli[i] + 'I'*(num_qubits-i-1)
        coeff = np.prod([sym_mean[j, gate_idx[pauli[j]]] for j in non_identity_idx if j != i])
        expanded.append(new_pauli)
        expanded_coeffs.append(coeff)
    
    if order >= 2:
        new_pauli = 'I'*num_qubits
        coeff = -(order-1) * np.prod([sym_mean[j, gate_idx[pauli[j]]] for j in non_identity_idx])
        expanded.append(new_pauli)
        expanded_coeffs.append(coeff)
    
    expanded_paulis = [qml.pauli.string_to_pauli_word(s, wire_map) for s in expanded]
    return expanded_paulis, expanded_coeffs

def compute_H_meanfield(H, num_qubits, sym_mean, wire_map=None):
    if wire_map is None:
        wire_map = dict(zip(range(num_qubits), range(num_qubits)))
        
    gate_idx = {'X':0, 'Y':1, 'Z':2}
    identity_str = 'I'*num_qubits
    
    coeffs = H.coeffs
    pauli_words = H.ops

    H_mf = np.zeros((num_qubits, 3))
    Id_coeff = 0
    
    for k,each in enumerate(pauli_words):
        expanded_paulis, expanded_coeffs = meanfield_reduction(num_qubits, pauli_word=each, sym_mean=sym_mean, wire_map=wire_map)
        for idx in range(len(expanded_paulis)):
            pauli = expanded_paulis[idx]
            pauli_str = qml.pauli.pauli_word_to_string(pauli, wire_map)
            
            if pauli_str != identity_str:
                wire = pauli.wires.labels[0] # always 1-qubit pauli
                #print((wire, gate_idx[pauli_str[wire]]))
                #print(expanded_coeffs[idx], coeffs[k])
                H_mf[wire, gate_idx[pauli_str[wire]]] += expanded_coeffs[idx] * coeffs[k]
            else:
                Id_coeff += expanded_coeffs[idx] * coeffs[k]
                
        #print('------')
    return H_mf, Id_coeff

def solve_1q_hamiltonian(a,b,c): # aX + bY + cZ
    matrix = np.array([[c,a-b*1j],[a+b*1j,-c]])
    eigvals, eigvecs = np.linalg.eigh(matrix)
    ground_state = eigvecs[:,0]
    
    X_mean = ground_state.conj().T @ np.array([[0,1],[1,0]]) @ ground_state
    Y_mean = ground_state.conj().T @ np.array([[0,-1j],[1j,0]]) @ ground_state
    Z_mean = ground_state.conj().T @ np.array([[1,0],[0,-1]]) @ ground_state
    
    #print('eigvals:', eigvals[0], eigvals[1])
    return np.real([X_mean, Y_mean, Z_mean]), eigvals[0], eigvals[1]


def meanfield_spectral_gap(num_qubits, H, qubit_means=None):
    
    if qubit_means is None:
        qubit_means = np.random.uniform(size=(num_qubits,3))

    for _ in range(20):
        H_mf, Id_coeff = compute_H_meanfield(H, num_qubits, sym_mean=qubit_means)
        total_ground_energy = 0
        smallest_qubit_spectral_gap = 1e8
        for q in range(num_qubits):
            a,b,c = H_mf[q,:]        
            new_means, qubit_ground_energy, qubit_first_energy = solve_1q_hamiltonian(a,b,c)
            #print(qubit_ground_energy, qubit_first_energy)
            qubit_spectral_gap = qubit_first_energy - qubit_ground_energy
            smallest_qubit_spectral_gap = min(smallest_qubit_spectral_gap, qubit_spectral_gap)

            qubit_means[q,:] = new_means
            total_ground_energy += qubit_ground_energy

        total_ground_energy += Id_coeff
        total_first_energy = total_ground_energy + smallest_qubit_spectral_gap
        total_spectral_gap = smallest_qubit_spectral_gap

    #print(total_ground_energy, total_first_energy)
    return total_spectral_gap


if __name__ == '__main__':
    num_qubits = 3
    paulis = ['III','XII','IYI','IIZ','ZXI','IYY','ZIY','XYZ']
    wire_map = dict(zip(range(num_qubits), range(num_qubits)))
    pauli_words = [qml.pauli.string_to_pauli_word(pauli, wire_map=wire_map) for pauli in paulis]

    coeffs = np.random.uniform(size=len(paulis))
    H = qml.Hamiltonian(coeffs, pauli_words)

    H_mat = qml.matrix(H,wire_order = range(num_qubits))
    eigvals, _ = np.linalg.eigh(H_mat)
    spectral_gap = eigvals[1] - eigvals[0]
    mf_gap = meanfield_spectral_gap(num_qubits, H)

    print('ground energy = ', eigvals[0])
    print('first energy = ', eigvals[1])
    print('spectral gap =', spectral_gap)
    print('meanfield gap =', mf_gap)