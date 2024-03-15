from utils import Element, Domain, BenchMarkValue, display_element_process
from typing import List, Callable
import random


def random_pop (neighborhood : List[Element]) -> Element:
    index = random.randint(0, len(neighborhood) - 1)
    return neighborhood.pop(index)




def tabu_simulated_annealing(f_cost : Callable[[Element], BenchMarkValue], domain : Domain, temperature : float, temp_decrease_factor : float,  tabu_length : int, k_max : int = 10):
    
    ########## Init variables #########
    
    # We start from a random point
    tabu_list = []
    S_best = domain.get_random_element()
    E_best = f_cost(S_best)
    L_neigh = domain.get_small_neighborhood(S_best)
    k = 0

    while k < k_max and len( L_neigh ) > 0:
        S = random_pop(L_neigh)
        if S.code in tabu_list :
            continue
        E = f_cost(S)
        if E - temperature < E_best :
            S_best = S
            L_neigh = domain.get_small_neighborhood(S_best)
            E_best = E
            tabu_list.append(S.code)
            if len(tabu_list) > tabu_length:
                tabu_list.pop(0)
        temperature *= temp_decrease_factor
        k+=1

    return S_best, E_best, k
    
    
    
    
    
def simulated_annealing(f_cost : Callable[[Element], BenchMarkValue], domain : Domain, temperature : float, temp_decrease_factor : float, k_max : int = 10):
    
    ########## Init variables #########
    
    # We start from a random point
    tabu_list = []
    S_best = domain.get_random_element()
    E_best = f_cost(S_best)
    L_neigh = domain.get_small_neighborhood(S_best)
    k = 0

    try :
        while k < k_max and len( L_neigh ) > 0:
            S = random_pop(L_neigh)
            if S.code in tabu_list :
                continue
            E = f_cost(S)
            if E - temperature < E_best :
                S_best = S
                L_neigh = domain.get_small_neighborhood(S_best)
                E_best = E
            temperature *= temp_decrease_factor
            k+=1
    except :
        print("Quit !")

    return S_best, E_best, k



def basic_hill_climbing (f_cost : Callable[[Element], BenchMarkValue], domain : Domain,  k_max : int = 10):
    
    ########## Init variables #########
    
    # We start from a random point
    S_best = domain.get_random_element()
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
        
        
    