import numpy as np
import pennylane as qml
from . import MeanFieldGap
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class Sampler:
    def __init__(self, num_qubits, pauli_terms, coeffs_t, tilde_coeffs_t, coeffs):
        self.num_qubits = num_qubits
        self.pauli_terms = pauli_terms
        self.x = coeffs_t
        self.y = tilde_coeffs_t
        self.z = coeffs
        self.d1 = np.linalg.norm(self.x-self.y)
        self.d2 = np.linalg.norm(self.x-self.z)
        self.d3 = np.linalg.norm(self.y-self.z)

        corners, axes, sizes = self.compute_bounding_box(self.y, self.d1, self.z, self.d2, c3=self.x, region='full')
        width, height = sizes
        self.area = width * height

    def max_gap_search(self, num_samples, dist_threshold=0.1, area_threshold=0.1):
        if self.d3 <= self.d1:
            ## Jump to final Hamiltonian
            #print('Final sampling')
            H = qml.Hamiltonian(self.z, self.pauli_terms)
            mf_gap = MeanFieldGap.meanfield_spectral_gap(self.num_qubits, H)
            return self.z, mf_gap, np.array([self.z])
        elif self.d1 <= dist_threshold:
            if self.d2 <= dist_threshold:
                ## Jump to final Hamiltonian
                #print('Final sampling')
                H = qml.Hamiltonian(self.z, self.pauli_terms)
                mf_gap = MeanFieldGap.meanfield_spectral_gap(self.num_qubits, H)
                return self.z, mf_gap, np.array([self.z])
            else:
                # RM = dist_threshold
                # Rm = RM/2.
                # start_theta = np.arcsin(self.height/(2*self.d2)) - np.pi/2
                # end_theta = start_theta + np.pi
                # samples = self.disk_sampling(self.x, Rm, RM, start_theta, end_theta, num_samples)

                print('Direct sampling')
                samples = self.intersection_sampling(num_samples, self.x, dist_threshold, self.z, self.d2, self.y, self.x, dist_threshold/2.)
                
        elif self.area <= area_threshold:
            # sampling in the disk (x,|x-y|)
            # RM = self.d1
            # Rm = RM/2.
            # start_theta = np.arcsin(self.height/(2*self.d2)) - np.pi/2
            # end_theta = start_theta + np.pi
            # samples = self.disk_sampling(self.x, Rm, RM, start_theta, end_theta, num_samples)

            print('Disk')
            ## Error 1/(2*d) * np.sqrt((4*d**2*r2**2) - (d**2-r1**2+r2**2)**2)
            samples = self.intersection_sampling(num_samples, self.x, self.d1, self.z, self.d2, self.y, self.x, self.d1/2.)
            
        else:
            print('Intersection')
            samples = self.intersection_sampling(num_samples, self.y, self.d1, self.z, self.d2, self.x, self.x, self.d1/2., 'full')
        
        samples_gap = []
        for i in range(len(samples)):
            H = qml.Hamiltonian(samples[i], self.pauli_terms)
            samples_gap.append(MeanFieldGap.meanfield_spectral_gap(self.num_qubits, H))

        max_gap_id = np.argmax(samples_gap)
        #assert np.linalg.norm(samples[max_gap_id] - coeffs) <= dist
        return samples[max_gap_id], samples_gap[max_gap_id], np.array(samples)
    
    def compute_bounding_box(self, c1, r1, c2, r2, c3=None, region='full'):
        ## Assume r1 < r2 and the disks have nonempyty intersection
        d = np.linalg.norm(c1-c2)
        unit_east = c2 - c1
        unit_east = unit_east / d
        if c3 is not None:
            dir = c3 - c1
            if np.linalg.norm(dir) < 1e-6:
                dir = np.random.rand(unit_east.shape[0])
                print("c3 must be different than c1,c2; otherwise take random direction.")
            
        else:
            dir = np.random.rand(unit_east.shape[0])

        unit_north = dir - (dir @ unit_east) * unit_east
        unit_north = unit_north / np.linalg.norm(unit_north)

        #print((4*d**2*r2**2) - (d**2-r1**2+r2**2)**2)
        intersect_r = 1/(2*d) * np.sqrt((4*d**2*r2**2) - (d**2-r1**2+r2**2)**2)
        a1 = np.sqrt(r1**2 - intersect_r**2)
        m = c1 + a1 * unit_east

        anchor_left = c2 - r2 * unit_east
        anchor_right = c1 + r1 * unit_east

        if d > r2: ## Case 1
            anchor_top = m + intersect_r * unit_north
            anchor_bot = m - intersect_r * unit_north
        elif d <= r2:
            anchor_top = c1 + r1 * unit_north
            anchor_bot = c2 - r1 * unit_north

        width = np.linalg.norm(anchor_right - anchor_left)
        height = np.linalg.norm(anchor_top - anchor_bot)

        if region == 'full':
            top_left = anchor_left + (height/2.) * unit_north
            bot_left = anchor_left - (height/2.) * unit_north
            top_right = anchor_right + (height/2.) * unit_north
            bot_right = anchor_right - (height/2.) * unit_north
        elif region == 'top-half':
            top_left = anchor_left + (height/2.) * unit_north
            bot_left = anchor_left
            top_right = anchor_right + (height/2.) * unit_north
            bot_right = anchor_right
        elif region == 'bot-half':
            top_left = anchor_left
            bot_left = anchor_left - (height/2.) * unit_north
            top_right = anchor_right
            bot_right = anchor_right - (height/2.) * unit_north
        elif region == 'left-half':
            top_left = anchor_left + (height/2.) * unit_north
            bot_left = anchor_left - (height/2.) * unit_north
            top_right = m + (height/2.) * unit_north
            bot_right = m - (height/2.) * unit_north
        elif region == 'right-half':
            top_left = m + (height/2.) * unit_north
            bot_left = m - (height/2.) * unit_north
            top_right = anchor_right + (height/2.) * unit_north
            bot_right = anchor_right - (height/2.) * unit_north
        
        corners = {'top_left':top_left, 'bot_left':bot_left, 'top_right':top_right, 'bot_right':bot_right}
        return corners, (unit_east, unit_north), (width, height)

    
    def intersection_sampling(self, num_samples, c1, r1, c2, r2, c3=None, repel_pt=None, repel_dist=None, region = 'full'):
        def sample(bot_left, width, height, unit_east, unit_north, num_samples):
            ax1 = np.random.rand(num_samples)
            ax2 = np.random.rand(num_samples)
            #return bot_left + np.outer(ax1,self.unit_east) + np.outer(ax2,self.unit_north)
            def compute_sample(i):
                return bot_left + ax1[i]*width * unit_east + ax2[i]*height * unit_north

            return np.array(Parallel(n_jobs=8)(delayed(compute_sample)(i) for i in range(num_samples)))

        def accept(samples):
            is_in_ball_1 = np.linalg.norm(samples - c1, axis=1) <= r1
            is_in_ball_2 = np.linalg.norm(samples - c2, axis=1) <= r2
            is_far_from_repel = np.linalg.norm(samples - repel_pt, axis=1) >= repel_dist
            all_satisfied = np.logical_and.reduce((is_in_ball_1, is_in_ball_2, is_far_from_repel), axis=0)
            return all_satisfied
        
        corners, axes, sizes = self.compute_bounding_box(c1, r1, c2, r2, c3=None, region='full')
        unit_east, unit_north = axes
        width, height = sizes

        out = sample(corners['bot_left'], width, height, unit_east, unit_north, num_samples)
        mask = accept(out)
        reject, = np.where(~mask)
        while reject.size > 0:
            fill = sample(corners['bot_left'], width, height, unit_east, unit_north, reject.size)
            mask = accept(fill)
            out[reject[mask]] = fill[mask]
            reject = reject[~mask]
        return out
        

    def disk_sampling(self, origin, Rm, RM, start_theta, end_theta, num_samples):
        ## Rm: inner radius, RM: outer radius
        unif_var_r = np.random.rand(num_samples)
        r = np.sqrt(unif_var_r * (RM*RM-Rm*Rm) + Rm*Rm)
        theta = np.random.uniform(start_theta, end_theta, num_samples)
        xs_coor = r*np.cos(theta)
        ys_coor = r*np.sin(theta)


        samples = [origin + xs_coor[i]*self.unit_east + ys_coor[i]*self.unit_north for i in range(num_samples)]
        return np.array(samples)
    
    def segment_sampling(self, u1, u2, num_samples, use_grid=False):
        if use_grid:
            s = np.linspace(0,1,num_samples)
        else:
            s = np.random.rand(num_samples)

        samples = [s[i] * u1 + (1-s[i]) * u2 for i in range(num_samples)] 
        return np.array(samples)
    
# def max_gap_search(num_qubits, pauli_terms, coeffs_t, tilde_coeffs_t, coeffs, use_grid=True):
#     x = coeffs_t
#     y = tilde_coeffs_t
#     z = coeffs
#     d1 = np.linalg.norm(x-y)
#     d2 = np.linalg.norm(x-z)
#     d3 = np.linalg.norm(y-z)

#     if np.linalg.norm(y-x) < 1e-1:
#         ## direct sampling
#         print('Sampling type 1')
#         u1 = x + (z-x)/15
#         u2 = x + (z-x)/5
#         samples = segment_sampling(u1,u2,num_samples=10,use_grid=True)
#     else:
#         print('Sampling type 2')
#         m = y + ((x-y) @ (z-y)) * (z-y)/d3**2
#         #u1 = 2*m - x
#         u1 = m + (m-x)/2
#         #u2 = y + (d1/d3) * (z-y)
#         u2 = m
#         v3 = z + (d2/d3) * (y-z)
#         u3 = (m+v3)/2.
#         samples = triangle_sampling(u1,u2,u3,num_samples_per_dim=10,use_grid=True)

#     if use_grid:
#         samples = np.unique(samples, axis=0)

#     samples_gap = []
#     for i in range(len(samples)):
#         H = qml.Hamiltonian(samples[i], pauli_terms)
#         samples_gap.append(MeanFieldGap.meanfield_spectral_gap(num_qubits, H))

#     max_gap_id = np.argmax(samples_gap)
#     assert np.linalg.norm(samples[max_gap_id] - coeffs) <= d2 + 1e-6
#     return samples[max_gap_id], samples_gap[max_gap_id], np.array(samples)

# def triangle_sampling(u1, u2, u3, num_samples_per_dim, use_grid=False):
#     # Shift coordinate u1 to 0
#     v2 = u2 - u1
#     v3 = u3 - u1
#     samples = []

#     if use_grid:
#         s = np.linspace(0,1,num_samples_per_dim)
#         t = np.linspace(0,1,num_samples_per_dim)
#     else:
#         s = np.random.rand(num_samples_per_dim)
#         t = np.random.rand(num_samples_per_dim)

#     ss, tt = np.meshgrid(s, t)
#     st = np.vstack([ss.ravel(), tt.ravel()]) # dim = 2 x num_samples
#     in_triangle = (np.sum(st, axis=0) <= 1)
#     samples = [st[0,i]*v2 + st[1,i]*v3 if in_triangle[i] else (1-st[0,i])*v2 + (1-st[1,i])*v3 for i in range(len(in_triangle))]
#     samples = u1 + np.array(samples)

#     return samples

# def segment_sampling(u1, u2, num_samples, use_grid=False):
#     if use_grid:
#         s = np.linspace(0,1,num_samples)
#     else:
#         s = np.random.rand(num_samples)

#     samples = [s[i] * u1 + (1-s[i]) * u2 for i in range(num_samples)] 
#     return samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num_qubits = 3
    pauli_strings = ['XII', 'XIZ']
    pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]

    coeffs_t = np.array([2,2])
    tilde_coeffs_t = np.array([1,1])
    coeffs = np.array([5,5])

    # coeffs_t = coeffs_t / np.linalg.norm(coeffs_t)
    # tilde_coeffs_t = tilde_coeffs_t / np.linalg.norm(tilde_coeffs_t)
    # coeffs = coeffs / np.linalg.norm(coeffs)

    #_,_,samples = max_gap_search(3, pauli_terms, coeffs_t,  tilde_coeffs_t, coeffs)
    sampler = Sampler(num_qubits, pauli_terms, coeffs_t,  tilde_coeffs_t, coeffs)

    #samples = np.array(triangle_sampling(coeffs_t, tilde_coeffs_t, coeffs))

    fig, ax = plt.subplots()
    _,_,samples = sampler.max_gap_search(num_samples=1000)
    # RM = np.linalg.norm(coeffs_t - tilde_coeffs_t)
    # Rm = RM/3.
    
    # start_theta = np.arcsin(sampler.height/(2*sampler.d2))
    # end_theta = start_theta + np.pi/2
    # print(Rm, RM, start_theta, end_theta)
    # samples = sampler.disk_sampling(sampler.x, Rm, RM, start_theta, end_theta, num_samples=1000)

    x = samples[:,0]
    y = samples[:,1]

    plt.scatter(x,y)
    plt.scatter([coeffs_t[0]], [coeffs_t[1]], label='c(t)')
    plt.scatter([tilde_coeffs_t[0]], [tilde_coeffs_t[1]], label="c'(t)")
    plt.scatter([coeffs[0]], [coeffs[1]], label='c')

    plt.plot([coeffs_t[0], tilde_coeffs_t[0]], [coeffs_t[1], tilde_coeffs_t[1]])
    plt.plot([coeffs_t[0], coeffs[0]], [coeffs_t[1], coeffs[1]])
    plt.plot([coeffs[0], tilde_coeffs_t[0]], [coeffs[1], tilde_coeffs_t[1]])

    circ1 = plt.Circle(tilde_coeffs_t, sampler.d1, color='r',fill=False)
    circ2 = plt.Circle(coeffs, sampler.d2, color='r',fill=False)

    ax.add_patch(circ1)
    ax.add_patch(circ2)

    plt.legend()
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()




