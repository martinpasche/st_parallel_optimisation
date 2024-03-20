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



def basic_hill_climbing (f_cost : Callable[[Element], BenchMarkValue], domain : Domain, S_init : Element = None,  k_max : int = 10):
    
    ########## Init variables #########
    
    # We start from a random point
    
    if S_init:
        S_best = S_init
    else: 
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
        




def VNS(f_cost, domain : Domain, HC_max_iter, n_max, global_max_iter, tabu_list_size):
  
  S = domain.get_random_element()
  BestScore = f_cost(S)
  print("BestScore = ", BestScore)
  global_iter = 0
  while global_iter<global_max_iter:
    n = 1
    current_score = BestScore
    previous_neighbors = []
    previously_visited = []
    while n < n_max:
      # Shake: Get an element from the neighborhood extended by n
      print("S: ", S)
      neighbors  = domain.VNS_neighborhood(S, previous_neighbors, n)
      
      for neighbor in neighbors:
        if neighbor not in previous_neighbors:
            previous_neighbors.append(neighbor)
      
      S_prime = random.choice(neighbors)
      a = 0
      while S_prime in previously_visited and a<7:
        S_prime = random.choice(neighbors)
        a += 1
      if len(previously_visited)<tabu_list_size:
        for element in domain.VNS_neighborhood(S):
          previously_visited.append(element)
      else:
        for element in domain.VNS_neighborhood(S):
          previously_visited.pop()
          previously_visited.append(element)
        
      print("cost_function = ", f_cost(S_prime))
      if f_cost(S_prime) < BestScore*0.62:
        #Search: Find the best in the neighborhood of S_prime
        S_prime_best, Local_BestScore, _ = basic_hill_climbing(f_cost, domain, S_prime, HC_max_iter)
      else:
        Local_BestScore = 0
      print("Local_BestScore = ", Local_BestScore)
      if Local_BestScore < BestScore:
        S = S_prime_best
        BestScore = Local_BestScore
        n = 1
        previous_neighbors = []
        print("New Best", BestScore)
      else:
        n += 1
        print("n", n)
    if BestScore == current_score:
      break
    else:
      global_iter += 1
      print("global_iter", global_iter)
          
  return S, BestScore, global_iter