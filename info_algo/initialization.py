import numpy as np

def initialization(nP, dim, ub, lb):
    Boundary_no = np.size(ub)
    
    # If the boundaries of all variables are equal and user enter a single number for both ub and lb
    if Boundary_no == 1:
        X = np.random.rand(nP, dim) * (ub - lb) + lb
    
    # If each variable has a different lb and ub
    elif Boundary_no > 1:
        X = np.zeros((nP, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = np.random.rand(nP) * (ub_i - lb_i) + lb_i
            
    return X
