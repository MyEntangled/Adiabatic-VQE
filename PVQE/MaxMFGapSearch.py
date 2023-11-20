import numpy as np
import pennylane as qml
import MeanFieldGap
import matplotlib.pyplot as plt

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
        ## Move right to z if d3 <= d1

        ## Variables named conditioned on x below the line yz.
        self.m = self.y + ((self.x-self.y) @ (self.z-self.y)) * (self.z-self.y)/self.d3**2
        self.anchor_bot = self.x
        self.anchor_top = 2*self.m -self.x
        self.anchor_left = self.z + (self.d2/self.d3) * (self.y-self.z)
        self.anchor_right = self.y + (self.d1/self.d3) * (self.z-self.y)
        
        self.unit_east = self.anchor_right - self.anchor_left
        if np.linalg.norm(self.unit_east) < 1e-6:
            self.unit_east = (self.z - self.y)/self.d3
        else:
            self.unit_east = self.unit_east / np.linalg.norm(self.unit_east)

        self.unit_north = self.anchor_top - self.anchor_bot
        if np.linalg.norm(self.unit_north) < 1e-6:
            random = np.random.rand(self.unit_east.shape[0])
            self.unit_north = random - (random @ self.unit_east) * self.unit_east
            self.unit_north = self.unit_north / np.linalg.norm(self.unit_north)
        else:
            self.unit_north = self.unit_north / np.linalg.norm(self.unit_north)

        self.far_top = self.y + self.d1*self.unit_north
        self.far_bot = self.y - self.d1*self.unit_north

        if np.linalg.norm(self.far_top - self.z) < self.d2:
            self.anchor_top = self.far_top
            self.anchor_bot = self.far_bot
            self.height = 2 * self.d1
        else:
            self.height = np.linalg.norm(self.anchor_top - self.anchor_bot)

        self.width = np.linalg.norm(self.anchor_left - self.anchor_right)
        self.area = self.width * self.height


    def max_gap_search(self, num_samples, dist_threshold=0.1, area_threshold=0.1):
        if self.d3 <= self.d1:
            print('Final sampling')
            H = qml.Hamiltonian(self.z, self.pauli_terms)
            mf_gap = MeanFieldGap.meanfield_spectral_gap(self.num_qubits, H)
            return self.z, mf_gap, np.array([self.z])
        elif self.d1 <= dist_threshold:
            # u1 = self.x + (self.z-self.x)/15
            # u2 = self.x + (self.z-self.x)/5
            # samples = self.segment_sampling(u1,u2,num_samples,use_grid=False)
            # print('Segment')
            if self.d2 <= dist_threshold:
                print('Final sampling')
                H = qml.Hamiltonian(self.z, self.pauli_terms)
                mf_gap = MeanFieldGap.meanfield_spectral_gap(self.num_qubits, H)
                return self.z, mf_gap, np.array([self.z])
            
            RM = dist_threshold
            Rm = RM/2.
            start_theta = np.arcsin(self.height/(2*self.d2)) - np.pi/2
            end_theta = start_theta + np.pi
            samples = self.disk_sampling(self.x, Rm, RM, start_theta, end_theta, num_samples)
            print('Direct sampling')
        elif self.area <= area_threshold:
            # sampling in the disk (x,|x-y|)
            RM = self.d1
            Rm = RM/2.
            start_theta = np.arcsin(self.height/(2*self.d2)) - np.pi/2
            end_theta = start_theta + np.pi
            samples = self.disk_sampling(self.x, Rm, RM, start_theta, end_theta, num_samples)
            print('Disk')
        else:
            samples = self.intersection_sampling(num_samples, 'full')
            print('Intersection')
        
        samples_gap = []
        for i in range(len(samples)):
            H = qml.Hamiltonian(samples[i], self.pauli_terms)
            samples_gap.append(MeanFieldGap.meanfield_spectral_gap(self.num_qubits, H))

        max_gap_id = np.argmax(samples_gap)
        #assert np.linalg.norm(samples[max_gap_id] - coeffs) <= dist
        return samples[max_gap_id], samples_gap[max_gap_id], np.array(samples)


    def intersection_sampling(self, num_samples, region = 'full'):
        def sample(bot_left, width, height, N):
            ax1 = np.random.rand(N)
            ax2 = np.random.rand(N)
            #return bot_left + np.outer(ax1,self.unit_east) + np.outer(ax2,self.unit_north)
            return np.array([bot_left + ax1[i]*width * self.unit_east + ax2[i]*height * self.unit_north for i in range(N)])

        def accept(samples):
            is_in_y_disk = np.linalg.norm(samples - self.y, axis=1) < self.d1 
            is_in_z_disk = np.linalg.norm(samples - self.z, axis=1) < self.d2
            is_in_intersection = np.logical_and(is_in_y_disk, is_in_z_disk)
            return is_in_intersection
        
        if region == 'full':
            top_left = self.anchor_left + (self.height/2)*self.unit_north
            bot_left = self.anchor_left - (self.height/2)*self.unit_north
            top_right = self.anchor_right + (self.height/2)*self.unit_north
            bot_right = self.anchor_right - (self.height/2)*self.unit_north
            area = self.area
            height = self.height
            width = self.width
            
        elif region == 'half':
            top_left = self.anchor_left + (self.height/2)*self.unit_north
            bot_left = self.anchor_left
            top_right = self.anchor_right + (self.height/2)*self.unit_north
            bot_right = self.anchor_right
            area = self.area/2.
            height = self.height/2
            width = self.width

        out = sample(bot_left, width, height, num_samples)
        mask = accept(out)
        reject, = np.where(~mask)
        while reject.size > 0:
            fill = sample(bot_left, width, height, reject.size)
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
    


def max_gap_search(num_qubits, pauli_terms, coeffs_t, tilde_coeffs_t, coeffs, use_grid=True):
    x = coeffs_t
    y = tilde_coeffs_t
    z = coeffs
    d1 = np.linalg.norm(x-y)
    d2 = np.linalg.norm(x-z)
    d3 = np.linalg.norm(y-z)

    if np.linalg.norm(y-x) < 1e-1:
        ## direct sampling
        print('Sampling type 1')
        u1 = x + (z-x)/15
        u2 = x + (z-x)/5
        samples = segment_sampling(u1,u2,num_samples=10,use_grid=True)
    else:
        print('Sampling type 2')
        m = y + ((x-y) @ (z-y)) * (z-y)/d3**2
        #u1 = 2*m - x
        u1 = m + (m-x)/2
        #u2 = y + (d1/d3) * (z-y)
        u2 = m
        v3 = z + (d2/d3) * (y-z)
        u3 = (m+v3)/2.
        samples = triangle_sampling(u1,u2,u3,num_samples_per_dim=10,use_grid=True)

    if use_grid:
        samples = np.unique(samples, axis=0)

    samples_gap = []
    for i in range(len(samples)):
        H = qml.Hamiltonian(samples[i], pauli_terms)
        samples_gap.append(MeanFieldGap.meanfield_spectral_gap(num_qubits, H))

    max_gap_id = np.argmax(samples_gap)
    assert np.linalg.norm(samples[max_gap_id] - coeffs) <= d2 + 1e-6
    return samples[max_gap_id], samples_gap[max_gap_id], np.array(samples)


# def max_gap_search(num_qubits, pauli_terms, coeffs_t, tilde_coeffs_t, coeffs, use_grid=True):
#     x = coeffs_t
#     y = tilde_coeffs_t
#     z = coeffs
#     gamma = (y - x) @ (z - x)
#     dist = np.linalg.norm(z-x)
#     if dist < 1e-2:
#         return z, 'Problem H'

#     if np.linalg.norm(y-x) < 1e-1:
#         ## direct sampling
#         print('Sampling type 1')
#         u1 = x + (z-x)/15
#         u2 = x + (z-x)/5
#         samples = segment_sampling(u1,u2,num_samples=10,use_grid=True)

#     else:
#         if gamma > 0:
#             ## triangle sampling
#             print('Sampling type 2')
#             u1 = y
#             u2 = x + (gamma/dist**2)*(z-x)
#             u3 = 2*u2 -x
#             samples = triangle_sampling(u1,u2,u3,num_samples_per_dim=10,use_grid=True)
#         else:
#             ## line segment sampling
#             print('Sampling type 3')
#             u1 = y + ((x-y) @ (z-y)) * (z-y)/np.linalg.norm(z-y)**2
#             u2 = 2*u1 - x
#             samples = segment_sampling(u1,u2,num_samples=10,use_grid=True)

#     if use_grid:
#         samples = np.unique(samples, axis=0)

#     samples_gap = []
#     for i in range(len(samples)):
#         H = qml.Hamiltonian(samples[i], pauli_terms)
#         samples_gap.append(MeanFieldGap.meanfield_spectral_gap(num_qubits, H))

#     max_gap_id = np.argmax(samples_gap)
#     #assert np.linalg.norm(samples[max_gap_id] - coeffs) <= dist
#     return samples[max_gap_id], samples_gap[max_gap_id], np.array(samples)

def triangle_sampling(u1, u2, u3, num_samples_per_dim, use_grid=False):
    # Shift coordinate u1 to 0
    v2 = u2 - u1
    v3 = u3 - u1
    samples = []

    if use_grid:
        s = np.linspace(0,1,num_samples_per_dim)
        t = np.linspace(0,1,num_samples_per_dim)
    else:
        s = np.random.rand(num_samples_per_dim)
        t = np.random.rand(num_samples_per_dim)

    ss, tt = np.meshgrid(s, t)
    st = np.vstack([ss.ravel(), tt.ravel()]) # dim = 2 x num_samples
    in_triangle = (np.sum(st, axis=0) <= 1)
    samples = [st[0,i]*v2 + st[1,i]*v3 if in_triangle[i] else (1-st[0,i])*v2 + (1-st[1,i])*v3 for i in range(len(in_triangle))]
    samples = u1 + np.array(samples)

    # for _ in range(num_samples):
    #     s,t = np.random.rand(2)
    #     in_triangle = (s + t <= 1)
    #     p = s * v2 + t * v3 if in_triangle else (1 - s) * v2 + (1 - t) * v3 
    #     samples.append(u1+p)
    return samples

def segment_sampling(u1, u2, num_samples, use_grid=False):
    if use_grid:
        s = np.linspace(0,1,num_samples)
    else:
        s = np.random.rand(num_samples)

    samples = [s[i] * u1 + (1-s[i]) * u2 for i in range(num_samples)] 
    return samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pauli_strings = ['XII', 'XIZ']
    #pauli_terms = [qml.pauli.string_to_pauli_word(str(each)) for each in pauli_strings]

    coeffs_t = np.array([0,2.6])

    tilde_coeffs_t = np.array([-1,2])

    coeffs = np.array([5,5])
    #_,_,samples = max_gap_search(3, pauli_terms, coeffs_t,  tilde_coeffs_t, coeffs)
    sampler = Sampler(coeffs_t,  tilde_coeffs_t, coeffs)

    #samples = np.array(triangle_sampling(coeffs_t, tilde_coeffs_t, coeffs))

    fig, ax = plt.subplots()
    samples = sampler.max_gap_search(num_samples=1000)
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




