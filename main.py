from utils import Element, Domain, BenchMarkValue, cost_function_benchmark, display_element_process, display_results, get_mark_max_gflops, get_mark_min_temp
from optim_algo import basic_hill_climbing, tabu_simulated_annealing


from mpi4py import MPI
import mpi4py

# MPI variables for the communications inter-processes
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

parameters = {
    "is_display" : False,
    "process" : Me,
    "folder_path" : "iso3dfd-st7",
}


#####################################################################
############################ INIT PROBLEM ###########################
#####################################################################

# We define the domain of the problema
Olevel_list = ["-O3", "-Ofast"]
simd_list = ["avx", "avx2", "avx512", "sse"]
problem_size_list1 = [256]
problem_size_list2 = [256]
problem_size_list3 = [256]
cache1_list = list(range(16, 257, 16))
cache2_list = list(range(1, 65, 1))
cache3_list = list(range(1, 65, 1))
iterations_list = [100]
threads_list = list(range(1, 65))

if Me == 0:
    print("Domain:\n")
    print("O level:", Olevel_list)
    print("Simd:", simd_list)
    print("Problem size 1:", problem_size_list1)
    print("Problem size 2:", problem_size_list2)
    print("Problem size 3:", problem_size_list3)
    print("Cache 1:", cache1_list)
    print("Cache 2:", cache2_list)
    print("Cache 3:", cache3_list)
    print("Iterations:", iterations_list)
    print("Threads:", threads_list)
    print("\n")



domain = Domain( Olevel_list, simd_list, problem_size_list1, problem_size_list2, problem_size_list3, cache1_list, cache2_list, cache3_list, threads_list, iterations_list)

#defining cost function for time, mpoints and gflops
def f_cost_time(S : Element) -> float:
    return cost_function_benchmark(S, **parameters).time
    
def f_cost_mpoints(S : Element) -> float:
    return cost_function_benchmark(S, **parameters).mpoints

def f_cost_gflops(S : Element) -> float:
    return -cost_function_benchmark(S, **parameters).flops


#####################################################################
############################### PROBLEM #############################
#####################################################################

if Me == 0:
    print("Tablu simulated annealing \n")
S_best, E_best, k = tabu_simulated_annealing(f_cost_gflops, domain, temperature = 100, temp_decrease_factor= 0.95, tabu_length = 20, k_max = 400)

mark = cost_function_benchmark(S_best, **parameters)
mark.k = k


#####################################################################
############################ PARALLEL CODE ##########################
#####################################################################

marks = comm.gather(mark, root=0)

if marks:
    # Display marks
    display_results(marks)
    mark_min_time = get_mark_min_temp(marks)
    mark_max_gflops = get_mark_max_gflops(marks)
    
    print("")
    print("Best time:\t\t", mark_min_time.element, mark_min_time)
    print("Highest GFlops:\t", mark_max_gflops.element, mark_max_gflops)