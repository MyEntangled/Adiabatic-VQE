import pennylane as qml
import numpy as np
#import pennylane.numpy as np
import numpy.random as random

from PVQE import AnsatzGenerator, LocalObservables, OrdinaryVQE, InitialParameters, MaxMFGapSearch, QWCGrouping, helper

class VQESolver():
    def __init__(self, num_qubits, pauli_strings, coeffs, ansatz_kwargs, start_ansatz_kwargs) -> None:
        self.num_qubits = num_qubits
        assert self.num_qubits == len(pauli_strings[0])
        self.wire_map = dict(zip(range(num_qubits), range(num_qubits)))

        ## Problem Hamiltonian
        self.pauli_strings = pauli_strings
        self.pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]

        self.coeffs_norm = np.linalg.norm(coeffs)
        self.coeffs = coeffs / self.coeffs_norm

        self.num_terms = len(self.pauli_strings)
        assert self.num_terms == len(self.pauli_terms)
        assert self.num_terms == len(coeffs)

        self.H = qml.Hamiltonian(self.coeffs, self.pauli_terms)
        #print(self.H)

        ## QWC grouping - List ordering
        self.term_groups = qml.pauli.group_observables(self.pauli_terms, grouping_type='qwc', method='rlf')
        self.grouping_to_list_map, self.list_to_grouping_map = self.qwc_grouping_structure()
        
        ## Initial Hamiltonian
        self.H0 = self.get_first_Hamiltonian()

        # Ansatz
        #self.ansatz_kwargs = None
        #self.num_parameters = None

        self.ansatz_kwargs = ansatz_kwargs
        self.start_ansatz_kwargs = start_ansatz_kwargs
        self.num_parameters = AnsatzGenerator.SimpleAnsatz(self.num_qubits, self.ansatz_kwargs['num_layers']).num_parameters
        self.theta_initializer = InitialParameters.ParameterInitializer(self.num_qubits, 
                                                                self.ansatz_kwargs['num_layers'],
                                                                self.ansatz_kwargs['ansatz_gen'],
                                                                dev=None)

        # Measurement terms
        self.prod_strings, self.prod_coeffs, self.new_measure_strings = LocalObservables.get_new_terms(pauli_strings)
        print('New strings:', len(self.new_measure_strings))
        print('Pauli strings:', len(self.pauli_strings))

        self.extended_strings = set(self.new_measure_strings).union(set(self.pauli_strings))
        self.extended_strings = list(self.extended_strings)
        print('All strings:', len(self.extended_strings))

        self.extended_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in self.extended_strings]
        print('All terms:', len(self.extended_terms))

        if len(self.extended_terms) <= 1000:
            self.extended_term_groups = qml.pauli.group_observables(self.extended_terms, grouping_type='qwc', method='rlf')
        else:
            self.extended_term_groups = QWCGrouping.grouping_pauli_terms(pauli_terms=self.extended_terms, 
                                                                         pauli_strings=self.extended_strings,
                                                                         wire_map=self.wire_map,
                                                                         random=True)

        self.extended_string_groups = [[qml.pauli.pauli_word_to_string(term, self.wire_map) for term in group] for group in self.extended_term_groups]
        # for i,string in enumerate(self.extended_strings):
        #     if string == 'ZIIIIIIIII':
        #         print(f'String {i} = ZIIIIIIIII')

        print(f'Number of extended terms = {len(self.extended_strings)}')

        # for i,group in enumerate(self.extended_string_groups):
        #     print(f'Group {i} has size {len(group)}')
        #     print(group)

    def qwc_grouping_structure(self):
        grouping_to_list_map = []
        list_to_grouping_map = [None]*len(self.pauli_terms)

        for i, group in enumerate(self.term_groups):
            temp = []
            for j, term in enumerate(group):
                in_list_id = -1
                for k in range(self.num_terms):
                    if term.compare(self.pauli_terms[k]):
                        in_list_id = k
                        break
                assert in_list_id != -1
                temp.append(in_list_id)
                list_to_grouping_map[in_list_id] = (i,j)
            grouping_to_list_map.append(temp)

        return grouping_to_list_map, list_to_grouping_map
    
    def grouping_to_list(self, grouping):
        lst = [None] * len(self.list_to_grouping_map)
        for i in range(len(lst)):
            group_id, member_id = self.list_to_grouping_map[i]
            lst[i] = grouping[group_id][member_id]
        return lst
    
    def list_to_grouping(self, lst):
        grouping = []
        for i, group in enumerate(self.grouping_to_list_map):
            temp = [lst[lst_id] for lst_id in group]
            grouping.append(temp)
        return grouping

        
    def get_first_Hamiltonian(self):
        H0_strings, _ = QWCGrouping.select_qwc_group(self.pauli_terms, self.pauli_strings, self.coeffs, criterion='max-l2-norm')
        print('H0 includes', H0_strings)
        H0_coeffs = np.array([self.coeffs[m] if self.pauli_strings[m] in H0_strings else 0 for m in range(self.num_terms)])
        H0 = qml.Hamiltonian(H0_coeffs, self.pauli_terms, grouping_type="qwc")
        return H0
    
    def get_genuine_hamiltonian(self, theta, coeffs):
        meas_dict = OrdinaryVQE.get_meas_outcomes(self.extended_term_groups, self.extended_string_groups, self.ansatz_kwargs, theta)

        M = LocalObservables.compute_M(meas_dict, self.pauli_strings, self.prod_strings, self.prod_coeffs)

        U, S, Vt = np.linalg.svd(M, full_matrices=True, compute_uv=True, hermitian=True)
        V = Vt.T

        #print(S)
        smallest_sv = S[-1]
        minspace_id = [id for id,v in enumerate(S) if v < smallest_sv + 1e-2]
        minspace_cols = V[:,minspace_id]
        inner_products = coeffs @ minspace_cols
        tilde_coeffs = np.sum(inner_products * minspace_cols, axis=1)

        return M, tilde_coeffs
            
    
    def solve(self, max_iters):
        ## Iterates...
        
        true_ge, true_fe = helper.true_ground_state_energy(self.H)
        true_ge *= self.coeffs_norm
        true_fe *= self.coeffs_norm
        record = {'H_prob': self.H*self.coeffs_norm, 'true_ge': true_ge, 'H_list':[], 'ground_energy':[], 'ground_theta':[], 'vqe_error':[]}

        coeffs_next = self.H0.coeffs
        for t in range(max_iters):
            print('Iteration: ', t+1)
            coeffs_curr = coeffs_next
            #print("Current coeffs:", coeffs_curr[:5])
            if t == 0:
                H_t = self.H0
                history = OrdinaryVQE.train_vqe(H_t, self.ansatz_kwargs, stepsize=0.1)
                ground_energy = history['energy'][-1]
                ground_theta = history['theta'][-1]

            else:
                H_t = qml.Hamiltonian(coeffs_curr, self.pauli_terms)
                coeff_group = self.list_to_grouping(coeffs_curr)

                # approx_ground_theta = self.theta_initializer.initialize(H_curr=H_t, theta_prev=ground_theta)

                #history = OrdinaryVQE.train_vqe((self.term_groups, coeff_group), self.ansatz_kwargs, init_theta = ground_theta)
                history = OrdinaryVQE.train_vqe(H_t, self.ansatz_kwargs, init_theta = ground_theta)

                # approx_history = OrdinaryVQE.train_vqe(H_t, self.ansatz_kwargs, approx_ground_theta)
                # energy = history['energy'][-1]
                # approx_energy = approx_history['energy'][-1]

                # if energy < approx_energy:
                #     print("Ground theta WINS:", len(approx_history['energy']) - len(history['energy']), np.round(approx_energy - energy))
                    
                #     ground_energy = energy
                #     ground_theta = history['theta'][-1]
                # else:
                #     print("Approx theta WINS:", len(approx_history['energy']) - len(history['energy']), np.round(approx_energy - energy))
                #     ground_energy = approx_energy
                #     ground_theta = approx_history['theta'][-1]           

                ground_energy = history['energy'][-1]
                ground_theta = history['theta'][-1]

            ## Genuine Hamiltonian
            ground_energy *= self.coeffs_norm
            print('End of VQE')
            M_t, tilde_coeffs = self.get_genuine_hamiltonian(theta = ground_theta, coeffs = coeffs_curr)
            #tilde_coeffs = tilde_coeffs * (np.inner(coeffs_curr, tilde_coeffs) / np.linalg.norm(tilde_coeffs)**2)

            #print('Correlation', M_t)
            ## Maximum Mean-Field Gap Search
            sampler = MaxMFGapSearch.Sampler(self.num_qubits, self.pauli_terms, coeffs_curr, tilde_coeffs, self.coeffs)
            coeffs_next, meanfield_gap, _ = sampler.max_gap_search(num_samples=100, dist_threshold=0.2, area_threshold=0.1)
            coeffs_next = coeffs_next
            
            #print('Tilde coeffs:', tilde_coeffs)
            true_ge, true_fe = helper.true_ground_state_energy(H_t)
            true_ge *= self.coeffs_norm
            true_fe *= self.coeffs_norm

            #anchor_ge, anchor_fe = helper.true_ground_state_energy(qml.Hamiltonian(tilde_coeffs, self.pauli_terms))

            print('True Energies: ', true_ge, true_fe)
            print('Est Ground Energy: ', ground_energy)
            #print('Anchor Energies', anchor_ge * self.coeffs_norm, anchor_fe * self.coeffs_norm)


            vqe_error = ground_energy - true_ge

            curr_to_H = np.linalg.norm(coeffs_curr - self.coeffs)
            next_to_H = np.linalg.norm(coeffs_next - self.coeffs)

            curr_to_anchor = np.linalg.norm(coeffs_curr - tilde_coeffs)
            anchor_to_next = np.linalg.norm(coeffs_next - tilde_coeffs)
            curr_to_next = np.linalg.norm(coeffs_next - coeffs_curr)

            ## Update record
            record['H_list'].append(H_t * self.coeffs_norm)
            record['ground_theta'].append(ground_theta)
            record['ground_energy'].append(ground_energy)
            record['vqe_error'].append(vqe_error)
            
            print('MF gap H(t+1):', meanfield_gap)
            print('Null-space deviation:', np.linalg.norm(M_t @ tilde_coeffs))
            print('VQE optimality:', np.linalg.norm(M_t @ coeffs_curr))
            print('VQE error:', vqe_error)
            print('H(t) H(t+1) to H:', np.round([curr_to_H, next_to_H], 3))
            print('H(t) to H̃(t):', np.round(curr_to_anchor, 3))
            print('H̃(t) to H(t+1):', np.round(anchor_to_next, 3))
            print('H(t) to H(t+1):', np.round(curr_to_next, 3))

            if np.linalg.norm(coeffs_curr - self.coeffs) < 1e-3:
                break
            print('----')

        return record


if __name__ == '__main__':
    # np.random.seed(10)
    # num_qubits = 12
    # num_layers = 3
    # full_basis = LocalObservables.get_k_local_basis(num_qubits, 3)
    # num_terms = len(full_basis) // 12

    # coeffs = np.random.rand(num_terms)
    # coeffs = coeffs / np.linalg.norm(coeffs)
    # #coeffs = np.array([1,2,4,8])
    # #coeffs = coeffs / np.linalg.norm(coeffs)

    # pauli_strings = random.choice(full_basis, size=num_terms, replace=False)
    # #pauli_strings = ['XII', 'XIZ', 'YIX', 'ZZY']
    # print(pauli_strings)
    # pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]

    # start_ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':0}
    # ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':num_layers}
    # solver = VQESolver(num_qubits, pauli_strings, coeffs, ansatz_kwargs, start_ansatz_kwargs)
    # solver.solve(50)

    ## CHUA TEST CASE C(T) CACH XA C HON C~ CACH C.

    from pennylane import qchem
    symbols = ["Li", "H"]
    coordinates = np.array([0.0, 0., 0.0, 0.0, 0.0, 5.5])
    #symbols = ["O", "H", "H"]
    #coordinates = np.array([0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, -0.7572, -0.4692])
    H, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates, active_electrons=2, active_orbitals=4)
    print(num_qubits)
    # symbols = ["H", "H", "H"]
    # R = 1.2
    # coordinates = np.array([[0, 0, 0], [0, 0, R/0.529], [0, 0, 2*R/0.529]])
    # H, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates, charge=1)

    num_layers = 4

    coeffs = H.coeffs
    pauli_terms = H.ops
    wire_map = dict(zip(range(num_qubits), range(num_qubits)))
    pauli_strings = [qml.pauli.pauli_word_to_string(term,wire_map) for term in pauli_terms]
        
    start_ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':0}
    ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':num_layers}
    solver = VQESolver(num_qubits, pauli_strings, coeffs, ansatz_kwargs, start_ansatz_kwargs)
    record = solver.solve(50)