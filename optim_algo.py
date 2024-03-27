from utils import Element, Domain, BenchMarkValue, display_element_process
from typing import List, Callable
import numpy as np
import random


def random_pop (neighborhood : List[Element]) -> Element:
    index = random.randint(0, len(neighborhood) - 1)
    return neighborhood.pop(index)

def retrieveMethod(method) :
    if method == "hc" :
        return basic_hill_climbing
    elif method == "sa" :
        return simulated_annealing
    elif method == "tsa" :
        return tabu_simulated_annealing
    elif method == "vns" :
        return VNS
    elif method == "random" :
        return random_elements
    elif method == 'info' :
        return INFO

def random_elements(f_cost : Callable[[Element], BenchMarkValue], starting_point : Element, domain : Domain, temperature : float, temp_decrease_factor : float,  tabu_length : int, k_max : int, HC_max_iter, n_max) :
    elements = [domain.get_random_element() for _ in range(k_max)]
    return sorted([(elem, f_cost(elem), k_max) for elem in elements], key=lambda tup : tup[1])[0]

def tabu_simulated_annealing(f_cost : Callable[[Element], BenchMarkValue], starting_point : Element, domain : Domain, temperature : float, temp_decrease_factor : float,  tabu_length : int, k_max : int, HC_max_iter, n_max):
    
    ########## Init variables #########
    
    # We start from a random point
    tabu_list = []
    S_best = starting_point
    E_best = f_cost(S_best)
    L_neigh = domain.get_neighborhood(S_best)
    k = 0

    while k < k_max and len( L_neigh ) > 0:
        S = random_pop(L_neigh)
        if S.code in tabu_list :
            continue
        E = f_cost(S)
        if E - temperature < E_best :
            S_best = S
            L_neigh = domain.get_neighborhood(S_best)
            E_best = E
            tabu_list.append(S.code)
            if len(tabu_list) > tabu_length:
                tabu_list.pop(0)
        temperature *= temp_decrease_factor
        k+=1

    return S_best, E_best, k
    
    
    
    
    
def simulated_annealing(f_cost : Callable[[Element], BenchMarkValue], starting_point : Element, domain : Domain, temperature : float, temp_decrease_factor : float,  tabu_length : int, k_max : int, HC_max_iter, n_max):
    
    ########## Init variables #########
    
    # We start from a random point
    tabu_list = []
    S_best = starting_point
    E_best = f_cost(S_best)
    L_neigh = domain.get_neighborhood(S_best)
    k = 0

    try :
        while k < k_max and len( L_neigh ) > 0:
            S = random_pop(L_neigh)
            if S.code in tabu_list :
                continue
            E = f_cost(S)
            if E - temperature < E_best :
                S_best = S
                L_neigh = domain.get_neighborhood(S_best)
                E_best = E
            temperature *= temp_decrease_factor
            k+=1
    except :
        print("Quit !")

    return S_best, E_best, k

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

def is_valid_float(s):
    try:
        float(s)  # Try converting the string to float
        return True
    except ValueError:
        return False

def INFO(f_cost : Callable[[Element], BenchMarkValue], starting_point : Element, domain : Domain, temperature : float, temp_decrease_factor : float,  tabu_length : int, k_max : int, HC_max_iter, n_max, population_size=30):
    # Initialization

    #compute lb and ub depending on the domain
    lb = [domain.get_variables()[i][0] for i in range(len(domain.get_variables()))][2:]
    ub = [domain.get_variables()[i][-1] for i in range(len(domain.get_variables()))][2:]
    # lb = [0 for i in range(len(domain.get_variables()))][2:]
    # ub = [len(domain.get_variables()[i]) for i in range(len(domain.get_variables()))][2:]
    prefix = np.array([domain.get_variables()[i][0] for i in range(2)])
    fobj = lambda x : f_cost(domain.get_element([domain.get_variables()[i].index(int(float(el))//16*16 if i == 7 else (int(float(el)) if is_valid_float(el) else el)) for i,el in enumerate(np.concatenate((prefix,x), axis=0))]))
    MaxIt = k_max
    dim = 8
    nP = population_size
    early_stopping_max = 10
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
    for it in range(MaxIt//population_size):
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
            #New_Cost = fobj(New_X)
            New_Cost = fobj(New_X)
            
            if New_Cost < Cost[i]:
                X[i, :] = New_X
                Cost[i] = New_Cost
                M[i] = Cost[i]
                if Cost[i] < Best_Cost:
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]
                    early_stopping = 0

        if M_Best == Best_Cost :
            early_stopping += 1

        if early_stopping >= early_stopping_max :
            break
        
        # Determine the worst and better solution
        ind = np.argsort(Cost)
        Worst_X = X[ind[-1], :]
        Worst_Cost = Cost[ind[-1]]
        I = np.random.randint(2, 6)
        Better_X = X[ind[I], :]
        Better_Cost = Cost[ind[I]]
    
    return Best_X, Best_Cost, (it+1)*population_size

def BC(X, lb, ub):
    Flag4ub = X > ub
    Flag4lb = X < lb
    X = X * (~(Flag4ub | Flag4lb)) + ub * Flag4ub + lb * Flag4lb
    return X

def basic_hill_climbing (f_cost : Callable[[Element], BenchMarkValue], starting_point : Element, domain : Domain, temperature : float, temp_decrease_factor : float,  tabu_length : int, k_max : int, HC_max_iter, n_max):
    
    ########## Init variables #########
    
    # We start from a random point
    S_best = starting_point
    E_best = f_cost(S_best)
    L_neigh = domain.get_neighborhood(S_best)
    k = 0
    
    while k < k_max and len( L_neigh ) > 0:
        S_1 = random_pop(L_neigh)
        E_1 = f_cost(S_1)
        
        if E_1 < E_best:
            S_best = S_1
            E_best = E_1
            L_neigh = domain.get_neighborhood(S_best)
            
        k += 1
    
    return S_best, E_best, k
        
def VNS(f_cost, starting_point : Element, domain : Domain, temperature : float, temp_decrease_factor : float, tabu_list_size, global_max_iter, HC_max_iter, n_max):
  
    S = starting_point
    BestScore = f_cost(S)
    global_iter = 0
    while global_iter<global_max_iter:
        n = 1
        current_score = BestScore
        previously_visited = []
        while n < n_max:
            # Shake: Get an element from the neighborhood extended by n
            S_prime  = domain.VNS_neighbor(S, previously_visited,  n)
            
            #Idea: Exclude the points in S_prime's small neighborhood
            if len(previously_visited)<tabu_list_size:
                previously_visited.append(S_prime)
            else:
                previously_visited.pop()
                previously_visited.append(S_prime)
              
            if f_cost(S_prime) < BestScore*0.62:
                #Search: Find the best in the neighborhood of S_prime
                S_prime_best, Local_BestScore, _ = basic_hill_climbing(f_cost, S_prime, domain, temperature, temp_decrease_factor, tabu_list_size, global_max_iter, HC_max_iter, n_max)
            else:
                Local_BestScore = 0
            if Local_BestScore < BestScore:
                S = S_prime_best
                BestScore = Local_BestScore
                n = 1
            else:
              n += 1
        if BestScore == current_score:
            break
        else:
            global_iter += 1
              
    return S, BestScore, global_iter