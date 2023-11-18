import pennylane as qml
import pennylane.numpy as np
import numpy.random as random

import AnsatzGenerator, LocalObservables, OrdinaryVQE, InitialParameters, MaxMFGapSearch, QWCGrouping
import helper

class VQESolver():
    def __init__(self, num_qubits, pauli_strings, coeffs, ansatz_kwargs, start_ansatz_kwargs) -> None:
        self.num_qubits = num_qubits
        assert self.num_qubits == len(pauli_strings[0])
        self.wire_map = dict(zip(range(num_qubits), range(num_qubits)))

        ## Problem Hamiltonian
        self.pauli_strings = pauli_strings
        self.pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]
        self.coeffs = coeffs

        self.num_terms = len(self.pauli_strings)
        assert self.num_terms == len(self.pauli_terms)
        assert self.num_terms == len(coeffs)

        self.H = qml.Hamiltonian(self.coeffs, self.pauli_terms)
        print(self.H)

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
        self.num_parameters = AnsatzGenerator.SimpleAnsatz(self.num_qubits, num_layers).num_parameters
        self.theta_approximator = InitialParameters.VQEApproximator(self.num_qubits, 
                                                                self.ansatz_kwargs['num_layers'],
                                                                self.ansatz_kwargs['ansatz_gen'],
                                                                dev=None)

        # Measurement terms
        self.prod_strings, self.prod_coeffs, self.new_measure_strings = LocalObservables.get_new_terms(pauli_strings)
        print('New strings:', self.new_measure_strings)
        print('Pauli strings:', self.pauli_strings)

        self.extended_strings = set(self.new_measure_strings).union(set(self.pauli_strings))
        self.extended_strings = list(self.extended_strings)

        self.extended_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in self.extended_strings]
        self.extended_term_groups = qml.pauli.group_observables(self.extended_terms, grouping_type='qwc', method='rlf')
        self.extended_string_groups = [[qml.pauli.pauli_word_to_string(term, self.wire_map) for term in group] for group in self.extended_term_groups]

        print('All strings:', self.extended_string_groups)

    def qwc_grouping_structure(self):
        grouping_to_list_map = []
        list_to_grouping_map = [None]*len(self.pauli_terms)
        print(self.pauli_terms)
        print(self.term_groups)
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
        H0_coeffs = np.zeros(shape=(self.num_terms))

        H0_coeffs = np.array([self.coeffs[m] if self.pauli_strings[m] in H0_strings else 0 for m in range(self.num_terms)])
        H0_coeffs = H0_coeffs
        # for m in range(self.num_terms):
        #     if self.pauli_strings[m] in H0_strings:
        #         H0_coeffs[m] = self.coeffs[m]

        H0 = qml.Hamiltonian(H0_coeffs, self.pauli_terms, grouping_type="qwc")
        return H0
    
    def get_genuine_hamiltonian(self, theta, coeffs):
        #meas_dict = LocalObservables.get_meas_outcomes(self.all_strings, self.ansatz_kwargs, theta)
        #M = LocalObservables.compute_correlation_matrix(meas_dict, self.pauli_strings, self.product_strings, self.product_phases, is_weighted=False)
        meas_dict = OrdinaryVQE.get_meas_outcomes(self.extended_term_groups, self.extended_string_groups, self.ansatz_kwargs, theta)
       #print(meas_dict)
        M = LocalObservables.compute_M(meas_dict, self.pauli_strings, self.prod_strings, self.prod_coeffs)
        eigvals, eigvecs = np.linalg.eigh(M)
        nullspace_id = [id for id,v in enumerate(eigvals) if v < 1e-6]

        if len(nullspace_id) > 0: ## M is singular
            nullspace_cols = eigvecs[:,nullspace_id]
            inner_products = coeffs @ nullspace_cols
            tilde_coeffs = np.sum(inner_products * nullspace_cols, axis=1)
        else: ## M_is nonsingular
            smallest_eigval = eigvals[0]
            subspace_id = [id for id,v in enumerate(eigvals) if v < smallest_eigval+1e-6]
            subspace_cols = eigvecs[:,subspace_id]
            inner_products = coeffs @ subspace_cols
            tilde_coeffs = np.sum(inner_products * subspace_cols, axis=1)

        return M, tilde_coeffs
            
    
    def solve(self, max_iters):
        ## Iterates...
        coeffs_next = self.H0.coeffs
        training_record = {'dist-to-H':[], 'dist-to-prev': []}

        for t in range(max_iters):
            print('Iteration: ', t+1)
            coeffs_curr = coeffs_next
            print("Current coeffs:", coeffs_curr)
            if t == 0:
                H_t = self.H0
                history = OrdinaryVQE.train_vqe(H_t, self.ansatz_kwargs, stepsize=0.1)
                ground_energy = history['energy'][-1]
                ground_theta = history['theta'][-1]

            else:
                H_t = qml.Hamiltonian(coeffs_curr, self.pauli_terms)
                coeff_group = self.list_to_grouping(coeffs_curr)

                #approx_ground_theta = self.theta_approximator.update(H_prev, H_t, theta_opt_prev=ground_theta, theta_init_curr=ground_theta)

                history = OrdinaryVQE.train_vqe((self.term_groups, coeff_group), self.ansatz_kwargs, ground_theta)
                #approx_history = OrdinaryVQE.train_vqe(H_t, self.ansatz_kwargs, approx_ground_theta)
                #energy = history['energy'][-1]
                #approx_energy = approx_history['energy'][-1]

                # if energy < approx_energy:
                #     # print("Ground theta WINS")
                #     # print("Difference", approx_energy - energy)
                #     ground_energy = energy
                #     ground_theta = history['theta'][-1]
                # else:
                #     # print("Approx theta WINS")
                #     # print("Difference", energy - approx_energy)
                #     ground_energy = approx_energy
                #     ground_theta = approx_history['theta'][-1]           

                ground_energy = history['energy'][-1]
                ground_theta = history['theta'][-1]

            ## Genuine Hamiltonian
            print('End of VQE')
            M_t, tilde_coeffs = self.get_genuine_hamiltonian(theta = ground_theta, coeffs = coeffs_curr)
            tilde_coeffs = tilde_coeffs * (np.inner(coeffs_curr, tilde_coeffs) / np.linalg.norm(tilde_coeffs)**2)

            #print('Correlation', M_t)
            ## Maximum Mean-Field Gap Search
            #coeffs_next, meanfield_gap, _ = MaxMFGapSearch.max_gap_search(self.num_qubits, self.pauli_terms, coeffs_curr, tilde_coeffs, self.coeffs)
            sampler = MaxMFGapSearch.Sampler(self.num_qubits, self.pauli_terms, coeffs_curr, tilde_coeffs, self.coeffs)
            coeffs_next, meanfield_gap, _ = sampler.max_gap_search(num_samples=100)
            coeffs_next = coeffs_next
            
            print('Tilde coeffs:', tilde_coeffs)
            true_ge, true_fe = helper.true_ground_state_energy(H_t)
            anchor_ge, anchor_fe = helper.true_ground_state_energy(qml.Hamiltonian(tilde_coeffs, self.pauli_terms))
            print('True Energies: ', true_ge, true_fe)
            print('Est Ground Energy: ', ground_energy)
            print('Anchor Energies', anchor_ge, anchor_fe)


            vqe_error = ground_energy - true_ge

            curr_to_H = np.linalg.norm(coeffs_curr - self.coeffs)
            next_to_H = np.linalg.norm(coeffs_next - self.coeffs)

            curr_to_anchor = np.linalg.norm(coeffs_curr - tilde_coeffs)
            anchor_to_next = np.linalg.norm(coeffs_next - tilde_coeffs)
            curr_to_next = np.linalg.norm(coeffs_next - coeffs_curr)

            ## Update record
            # training_record['dist-to-H'].append(diff_to_H)
            # training_record['dist-to-prev'].append(diff_to_prev)
            
            print('next c', coeffs_next)
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


if __name__ == '__main__':
    # np.random.seed(10)
    # num_qubits = 8
    # num_layers = 3
    # full_basis = LocalObservables.get_k_local_basis(num_qubits, 3)
    # num_terms = len(full_basis) // 20

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
    # solver.solve(20)


    from pennylane import qchem
    symbols = ["H", "O", "H"]
    coordinates = np.array([-0.0399, -0.0038, 0.0, 1.5780, 0.8540, 0.0, 2.7909, -0.5159, 0.0])
    H, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates)
    num_layers = 4

    coeffs = H.coeffs
    pauli_terms = H.ops
    wire_map = dict(zip(range(num_qubits), range(num_qubits)))
    pauli_strings = [qml.pauli.pauli_word_to_string(term,wire_map) for term in pauli_terms]
            

    start_ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':0}
    ansatz_kwargs = {'ansatz_gen':"SimpleAnsatz", 'num_qubits':num_qubits, 'num_layers':num_layers}
    solver = VQESolver(num_qubits, pauli_strings, coeffs, ansatz_kwargs, start_ansatz_kwargs)
    solver.solve(50)