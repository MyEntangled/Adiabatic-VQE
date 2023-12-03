import pennylane as qml
import pennylane.numpy as np

from PVQE import AnsatzGenerator

import itertools
import sys

def get_k_local_basis(num_qubits, k):
    assert num_qubits >= k
    pauli_type = ['I','X','Y','Z']
    #gate_dict = {'X': X, 'Y': Y, 'Z': Z}
    L_list = []
    for n in range(num_qubits):
        all_local_ops = itertools.product(['I','X','Y','Z'], repeat=k)
        for local_op in all_local_ops:
            pauli_str = ['I'] * num_qubits
            for i,char in enumerate(local_op[::-1]):
                pauli_str[(n + i)%num_qubits] = char
            
            pauli_str = ''.join(pauli_str)
            if pauli_str != 'I'*num_qubits:
                L_list.append(pauli_str)
    return list(set(L_list))  


def get_hamiltonian_terms(num_qubits, H):
    gate_name = {'PauliX':'X', 'PauliY':'Y', 'PauliZ':'Z', 'I':'I', 'Identity':'I'}

    assert isinstance(H, qml.Hamiltonian)

    terms_lst = H.ops
    coeffs = H.coeffs
    num_terms = len(terms_lst)
    terms_str = []

    #wire_map = dict(zip(range(num_qubits), range(num_qubits)))

    for i in range(num_terms):
        term = terms_lst[i]
        name = term.name
        if not isinstance(name, list):
            name = [name]
        wires = term.wires.labels
        gate_str = list('I'*num_qubits)

        for j,wire in enumerate(wires):
            gate_str[wire] = gate_name[name[j]]
        terms_str.append(''.join(gate_str))
        
        #terms_str = qml.pauli.pauli_word_to_string(term, wire_map=wire_map)

    return terms_str, coeffs

def get_new_terms(pauli_strings):
    ## A,B: 2 Pauli strings
    ## AB+BA = 2AB if they are different at an even number of non-identity Pauli operators
    ## Otherwise AB+BA = 0
    prod_dict = {('X','Y'): ('Z', 1j), ('Y','X'): ('Z', -1j),
                 ('Y','Z'): ('X', 1j), ('Z','Y'): ('X', -1j),
                 ('Z','X'): ('Y', 1j), ('X','Z'): ('Y', -1j)}
    def anticommutator(A,B):
        prod_string = []
        coeff = 1
        count = 0
        for i in range(len(A)):
            if A[i] == 'I':
                prod_string.append(B[i])
            elif B[i] == 'I':
                prod_string.append(A[i])
            elif A[i] == B[i]:
                prod_string.append('I')
            else:
                count += 1
                prod, c = prod_dict[(A[i],B[i])]
                prod_string.append(prod)
                coeff = coeff * c

        if count % 2 == 1:
            return (None, 0)
        else:
            return (''.join(prod_string), 2*np.real(coeff)) ## the final coeff is always real.
    #--------------

    num_terms = len(pauli_strings)
    prod_strings = []
    prod_coeffs = []
    new_measure_strings = []

    for i in range(num_terms):
        strings = []
        coeffs = []
        for j in range(i+1, num_terms):
            anti_term, coeff = anticommutator(pauli_strings[i], pauli_strings[j])
            strings.append(anti_term)
            coeffs.append(coeff)
            if anti_term is not None:
                new_measure_strings.append(anti_term)

        prod_strings.append(strings)
        prod_coeffs.append(coeffs)
    return prod_strings, prod_coeffs, new_measure_strings

def get_cov_terms(num_qubits, H=None, pauli_strings=None, pauli_terms=None, coeffs=None):
    if H:
        pauli_strings, coeffs = get_hamiltonian_terms(num_qubits, H)
        pauli_terms = H.ops
    elif pauli_strings is not None and coeffs is not None:
        pass
    else:
        print('Please provide either Hamiltonian or Pauli terms and coefficients')

    cov_terms_str = []
    phases = []
    num_terms = len(pauli_terms)

    for i in range(num_terms):
        cov_temp = []
        phase_temp = []
        for j in range(num_terms):
            if j != i:
                cov, phase = qml.pauli.pauli_mult_with_phase(pauli_terms[i], pauli_terms[j])
                cov_str, phase = get_hamiltonian_terms(num_qubits, qml.Hamiltonian([phase], [cov]))
                cov_str = cov_str[0]
                phase = phase[0]
            if j == i:
                cov_str = 'I'*num_qubits
                phase = 1.
            
            cov_temp.append(cov_str)
            phase_temp.append(phase)

        cov_terms_str.append(cov_temp)
        phases.append(phase_temp)

    cov_coeffs = np.outer(coeffs, coeffs)
    return cov_terms_str, phases, cov_coeffs

def get_meas_outcomes(meas_str, ansatz_kwargs, theta, dev=None, H=None, H_mixer=None):
    # locals().update(ansatz_kwargs)
    # sys._getframe(1).f_locals.update(ansatz_kwargs)
    # print(set(locals()))
    # assert set(['ansatz_gen','num_qubits','num_layers']).issubset(set(locals()))
        
    num_qubits = ansatz_kwargs['num_qubits']
    num_layers = ansatz_kwargs['num_layers']
    ansatz_gen = ansatz_kwargs['ansatz_gen']
    if ansatz_gen == 'QAOAAnsatz':
        assert H is not None
        assert H_mixer is not None

    wire_map = dict(zip(range(num_qubits), range(num_qubits)))
    meas_terms = [qml.pauli.string_to_pauli_word(pauli_str, wire_map) for pauli_str in meas_str]
    
    if not dev:
        dev = qml.device('default.qubit', wires=num_qubits)

    ansatz_obj = getattr(AnsatzGenerator,ansatz_gen)(num_qubits, num_layers)

    @qml.qnode(dev, interface='autograd', diff_method='best')
    def circuit(theta, H=None, H_mixer=None):
        if ansatz_gen == 'QAOAAnsatz':
            gamma = theta[:ansatz_obj.num_layers]
            beta = theta[ansatz_obj.num_layers:]
            ansatz_obj.get_ansatz(gamma, beta, H, H_mixer)
        else:
            ansatz_obj.get_ansatz(theta)
        return [qml.expval(term) for term in meas_terms]
    
    if ansatz_gen != 'QAOAAnsatz':
        meas_outcomes = circuit(theta)
    else:
        meas_outcomes = circuit(theta, H, H_mixer)
    meas_dict = dict(zip(meas_str, meas_outcomes))
    return meas_dict

def compute_M(meas_dict, pauli_strings, prod_strings, prod_coeffs):
    num_terms = len(pauli_strings)

    M_anticomm = np.zeros((num_terms, num_terms))
    for i in range(num_terms):
        strings = prod_strings[i]
        coeffs = prod_coeffs[i]
        for pos, j in enumerate(range(i+1,num_terms)):
            anti_string = strings[pos]
            coeff = coeffs[pos]
            if anti_string is None:
                M_anticomm[i,j] = 0
            else:
                M_anticomm[i,j] = coeff * meas_dict[anti_string]
    M_anticomm = M_anticomm + M_anticomm.T
    np.fill_diagonal(M_anticomm, 2.)

    expvals = np.array([meas_dict[string] for string in pauli_strings])
    M_exp = np.outer(expvals, expvals)
    # print(expvals)
    # print(M_anticomm)
    # print(M_exp)
    M = (1./2) * M_anticomm - M_exp
    return M


def compute_correlation_matrix(meas_dict, pauli_strings, product_strings, product_phases, is_weighted=None, product_coeffs=None):
    num_terms = len(product_strings)

    exp_lst = np.array([meas_dict[exp_str] for exp_str in pauli_strings])

    cross_prod_mat = np.zeros((num_terms, num_terms))
    for i in range(num_terms):
        for j in range(num_terms):
            if i < j:
                temp = (meas_dict[product_strings[i][j]] * product_phases[i][j] + meas_dict[product_strings[j][i]] * product_phases[j][i]) / 2.
                cross_prod_mat[i,j] = np.real(temp)
            elif i > j:
                cross_prod_mat[i,j] = cross_prod_mat[j,i]
            else:
                cross_prod_mat[i,j] = 1

    #cross_prod_mat = cross_prod_mat * np.array(product_phases)
    cov_mat = cross_prod_mat - np.outer(exp_lst, exp_lst)

    if not is_weighted:
        assert(np.linalg.norm(cov_mat - (cov_mat + cov_mat.conj().T)/2.) < 1e-5)
        M = cov_mat + cov_mat.conj().T
    elif is_weighted:
        weighted_cov_mat = cov_mat * np.array(product_coeffs)
        M = weighted_cov_mat + weighted_cov_mat.conj().T

    return M/2.


if __name__ == '__main__':
    pauli_terms = ['XYZX', 'YYZX', 'YZZX','YZXX','YZXY']
    print(get_new_terms(pauli_terms))

    num_qubits = 3
    coeffs = [0.2, -0.543]
    obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliY(2)]
    H = qml.Hamiltonian(coeffs, obs)
    print(get_hamiltonian_terms(num_qubits, H))
    cov_terms, cov_phases, cov_coeffs = get_cov_terms(num_qubits, H)
    print(cov_terms)
    print(cov_phases)
    print(cov_coeffs)

    exp_terms_str, exp_coeffs = get_hamiltonian_terms(num_qubits, H)
    cov_terms_str, cov_phases, cov_coeffs = get_cov_terms(num_qubits, H)

    meas_str = set(itertools.chain.from_iterable(cov_terms_str)).union(exp_terms_str)
    meas_str = list(meas_str)

    ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':3}
    num_parameters = AnsatzGenerator.SimpleAnsatz(num_qubits, num_layers=3).num_parameters
    theta = np.random.rand(num_parameters, requires_grad=True)

    meas_dict = get_meas_outcomes(meas_str, ansatz_kwargs, theta)
    M = compute_correlation_matrix(meas_dict, exp_terms_str, cov_terms_str, cov_phases, is_weighted=False)
    print(M)