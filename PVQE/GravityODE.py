import numpy as np

#r = np.sqrt(2**ansatz.num_qubits)

def ode(t, z, H_coeffs, mass, prev_w, nullspace):
    l = len(z)//2
    w = z[:l]
    wdot = z[l:]
    d = np.linalg.norm(H_coeffs - w)
    
    ## Gravity towards H
    #g = mass * (H_coeffs - w) / d**3
    g = (H_coeffs - w) / d
    ## Gravity towards null_w
    null_w = (w @ nullspace) * nullspace
    
    #assert np.linalg.norm(M @ null_w) < 0.001
    h = (null_w - w)
    #h = 0.8 * h * mass  / (np.linalg.norm(h) * d**2)
    h = 0.5 * h / np.linalg.norm(h)
    
    #print("g vs h", np.linalg.norm(g), np.linalg.norm(h))
    wddot = g + h
    #    print('Attraction: ', np.linalg.norm(wddot), wddot @ g, wddot @ h)
    #wddot = wddot / np.linalg.norm(wddot)

    res = list(np.concatenate((wdot, wddot)))
    return res

# Define the event function
def converge_cond(t, z, H_coeffs, mass, prev_w, nullspace): 
    if np.linalg.norm(z[:len(z)//2] - H_coeffs) <= 0.001:
        return 0
    else:
        return 1

def deviation_bound(t, z, H_coeffs, mass, prev_w, nullspace):
    if np.linalg.norm(z[:len(z)//2] - prev_w) >= 0.1: #0.1/r:
        return 0
    else: return 1