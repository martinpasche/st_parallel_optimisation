import numpy as np
from initialization import initialization

def INFO(nP, MaxIt, lb, ub, dim, fobj):
    # Initialization
    Cost = np.zeros(nP)
    M = np.zeros(nP)
    
    X = initialization(nP, dim, ub, lb)
    
    for i in range(nP):
        Cost[i] = fobj(np.floor(X[i, :]))
        M[i] = Cost[i]
    
    ind = np.argsort(Cost)
    Best_X = X[ind[0], :]
    Best_Cost = Cost[ind[0]]
    
    Worst_Cost = Cost[ind[-1]]
    Worst_X = X[ind[-1], :]
    
    I = np.random.randint(2, 6)
    Better_X = X[ind[I], :]
    Better_Cost = Cost[ind[I]]
    
    # Main Loop of INFO
    Convergence_curve = np.zeros(MaxIt)
    for it in range(MaxIt):
        alpha = 2 * np.exp(-4 * (it / MaxIt))
        
        M_Best = Best_Cost
        M_Better = Better_Cost
        M_Worst = Worst_Cost
        
        for i in range(nP):
            # Updating rule stage
            del_ = 2 * np.random.rand() * alpha - alpha
            sigm = 2 * np.random.rand() * alpha - alpha
            
            # Select three random solutions
            A1 = np.random.permutation(nP)
            A1 = A1[A1 != i]
            a, b, c = A1[:3]
            
            e = 1e-25
            epsi = e * np.random.rand()
            
            omg = max(M[a], M[b], M[c])
            MM = np.array([M[a] - M[b], M[a] - M[c], M[b] - M[c]])
            
            W = np.cos(MM + np.pi) * np.exp(-MM / omg)
            Wt = np.sum(W)
            
            WM1 = del_ * (W[0] * (X[a, :] - X[b, :]) + W[1] * (X[a, :] - X[c, :]) +
                          W[2] * (X[b, :] - X[c, :])) / (Wt + 1) + epsi
            
            omg = max(M_Best, M_Better, M_Worst)
            MM = np.array([M_Best - M_Better, M_Best - M_Worst, M_Better - M_Worst])
            
            W = np.cos(MM + np.pi) * np.exp(-MM / omg)
            Wt = np.sum(W)
            
            WM2 = del_ * (W[0] * (Best_X - Better_X) + W[1] * (Best_X - Worst_X) +
                          W[2] * (Better_X - Worst_X)) / (Wt + 1) + epsi
            
            # Determine MeanRule
            r = np.random.uniform(0.1, 0.5)
            MeanRule = r * WM1 + (1 - r) * WM2
            
            if np.random.rand() < 0.5:
                z1 = X[i, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (Best_X - X[a, :]) / (M_Best - M[a] + 1)
                z2 =Best_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (M[a] - M[b] + 1)
            else:
                z1 = X[a, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[b, :] - X[c, :]) / (M[b] - M[c] + 1)
                z2 = Better_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (M[a] - M[b] + 1)
            
            # Vector combining stage
            u = np.zeros(dim)
            for j in range(dim):
                mu = 0.05 * np.random.randn()
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        u[j] = z1[j] + mu * abs(z1[j] - z2[j])
                    else:
                        u[j] = z2[j] + mu * abs(z1[j] - z2[j])
                else:
                    u[j] = X[i, j]
            
            # Local search stage
            if np.random.rand() < 0.5:
                L = np.random.rand() < 0.5
                v1 = (1 - L) * 2 * np.random.rand() + L
                v2 = np.random.rand() * L + (1 - L)
                Xavg = (X[a, :] + X[b, :] + X[c, :]) / 3
                phi = np.random.rand()
                Xrnd = phi * Xavg + (1 - phi) * (phi * Better_X + (1 - phi) * Best_X)
                Randn = L * np.random.randn(dim) + (1 - L) * np.random.randn()
                if np.random.rand() < 0.5:
                    u = Best_X + Randn * (MeanRule + np.random.randn() * (Best_X - X[a, :]))
                else:
                    u = Xrnd + Randn * (MeanRule + np.random.randn() * (v1 * Best_X - v2 * Xrnd))
            
            # Check if new solution goes outside the search space and bring them back
            New_X = np.floor(BC(u, lb, ub))
            New_Cost = fobj(New_X)
            
            if New_Cost < Cost[i]:
                X[i, :] = New_X
                Cost[i] = New_Cost
                M[i] = Cost[i]
                if Cost[i] < Best_Cost:
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]
        
        # Determine the worst and better solution
        ind = np.argsort(Cost)
        Worst_X = X[ind[-1], :]
        Worst_Cost = Cost[ind[-1]]
        I = np.random.randint(2, 6)
        Better_X = X[ind[I], :]
        Better_Cost = Cost[ind[I]]
        
        # Update Convergence_curve
        Convergence_curve[it] = Best_Cost
        
        # Show Iteration Information
        print(f'Iteration {it + 1}, Best Cost = {Best_Cost}')
    
    return Best_Cost, Best_X, Convergence_curve

def BC(X, lb, ub):
    Flag4ub = X > ub
    Flag4lb = X < lb
    X = X * (~(Flag4ub | Flag4lb)) + ub * Flag4ub + lb * Flag4lb
    return X
