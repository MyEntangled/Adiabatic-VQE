import numpy as np
import scipy


#r = np.sqrt(2**ansatz.num_qubits)

# def ode(t, z, H_coeffs, mass, prev_w, nullspace):
#     l = len(z)//2
#     w = z[:l]
#     wdot = z[l:]
#     d = np.linalg.norm(H_coeffs - w)
    
#     ## Gravity towards H
#     #g = mass * (H_coeffs - w) / d**3
#     g = (H_coeffs - w) / d
#     ## Gravity towards null_w
#     null_w = (w @ nullspace) * nullspace
    
#     #assert np.linalg.norm(M @ null_w) < 0.001
#     h = (null_w - w)
#     #h = 0.8 * h * mass  / (np.linalg.norm(h) * d**2)
#     h = 0.5 * h / np.linalg.norm(h)
    
#     #print("g vs h", np.linalg.norm(g), np.linalg.norm(h))
#     wddot = g + h
#     #    print('Attraction: ', np.linalg.norm(wddot), wddot @ g, wddot @ h)
#     #wddot = wddot / np.linalg.norm(wddot)

#     res = list(np.concatenate((wdot, wddot)))
#     return res

# # Define the event function
# def converge_cond(t, z, H_coeffs, mass, prev_w, nullspace): 
#     if np.linalg.norm(z[:len(z)//2] - H_coeffs) <= 0.001:
#         return 0
#     else:
#         return 1

# def deviation_bound(t, z, H_coeffs, mass, prev_w, nullspace):
#     if np.linalg.norm(z[:len(z)//2] - prev_w) >= 0.01: #0.1/r:
#         return 0
#     else: return 1

class SecondOrderAttraction():
    def __init__(self, t_span, H_coeffs):
        self.t_span = t_span
        self.H_coeffs = H_coeffs

    def ode(self, t, z, nullspace, prev_w):
        l = len(z)//2
        w = z[:l]
        wdot = z[l:]
        d = np.linalg.norm(self.H_coeffs - w)
        
        ## Gravity towards H
        g = (self.H_coeffs - w) / d
        ## Gravity towards null_w
        null_w = (w @ nullspace) * nullspace
        
        #assert np.linalg.norm(M @ null_w) < 0.001
        h = (null_w - w)
        #h = 0.8 * h * mass  / (np.linalg.norm(h) * d**2)
        h = 0.5 * h / np.linalg.norm(h)
        
        #print("g vs h", np.linalg.norm(g), np.linalg.norm(h))
        wddot = g + h

        res = list(np.concatenate((wdot, wddot)))
        return res
    
    # Define the event function
    def converge_cond(self, t, z, nullspace, prev_w): 
        if np.linalg.norm(z[:len(z)//2] - self.H_coeffs) <= 0.001:
            return 0
        else:
            return 1

    def deviation_bound(self, t, z, nullspace, prev_w):
        if np.linalg.norm(z[:len(z)//2] - prev_w) >= 0.05: #0.1/r:
            return 0
        else: return 1

    def solve_ivp(self, init_z, nullspace, prev_w, time_step):
        sol = scipy.integrate.solve_ivp(self.ode, self.t_span, init_z, events=(self.converge_cond, self.deviation_bound), 
                       max_step=time_step, args=(nullspace, prev_w))
        z = sol.y[:,-1]
        return z


class FirstOrderAttraction():
    def __init__(self, t_span, H_coeffs):
        self.t_span = t_span
        self.H_coeffs = H_coeffs
        # self.converge_cond.terminal = True
        # self.converge_cond.direction = -1
        # self.deviation_bound.terminal = True
        # self.deviation_bound.terminal = -1

    def ode(self, t, z, nullspace, prev_w):

        d = np.linalg.norm(self.H_coeffs - z)
        #print("time-dist: ", t, d)

        ## Gravity towards H
        g = (self.H_coeffs - z) / d
        ## Gravity towards null_w
        null_z = (z @ nullspace) * nullspace
        
        #assert np.linalg.norm(M @ null_w) < 0.001
        h = (null_z - z)
        #h = 0.8 * h * mass  / (np.linalg.norm(h) * d**2)
        h = h / np.linalg.norm(h)
        
        #print("g vs h", np.linalg.norm(g), np.linalg.norm(h))
        wdot = g + 0.5*h
        return wdot
    
    # Define the event function
    def converge_cond(self, t, z, nullspace, prev_w): 
        if np.linalg.norm(z - self.H_coeffs) <= 0.001:
            return 0
        else:
            return 1

    def deviation_bound(self, t, z, nullspace, prev_w):
        if np.linalg.norm(z - prev_w) >= 0.05: #0.1/r:
            return 0
        else: return 1

    def solve_ivp(self, init_z, nullspace, prev_w, time_step):
        sol = scipy.integrate.solve_ivp(self.ode, self.t_span, init_z, events=(self.converge_cond, self.deviation_bound), 
                       max_step=time_step, args=(nullspace, prev_w))
        z = sol.y[:,-1]
        return z



# def ode(t, z, H_coeffs, mass, prev_w, nullspace):
#     d = np.linalg.norm(H_coeffs - z)
    
#     ## Gravity towards H
#     g = (H_coeffs - z) / d
#     ## Gravity towards null_w
#     null_z = (z @ nullspace) * nullspace
    
#     #assert np.linalg.norm(M @ null_w) < 0.001
#     h = (null_z - z)
#     #h = 0.8 * h * mass  / (np.linalg.norm(h) * d**2)
#     h = h / np.linalg.norm(h)
    
#     #print("g vs h", np.linalg.norm(g), np.linalg.norm(h))
#     wdot = g + 0.5*h
#     return wdot

# # Define the event function
def converge_cond(t, z, H_coeffs, mass, prev_w, nullspace): 
    if np.linalg.norm(z - H_coeffs) <= 0.001:
        return 0
    else:
        return 1

def deviation_bound(t, z, H_coeffs, mass, prev_w, nullspace):
    if np.linalg.norm(z - prev_w) >= 0.01: #0.1/r:
        return 0
    else: return 1