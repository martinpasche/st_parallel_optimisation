from utils import Element, Domain, BenchMarkValue, cost_function_benchmark, display_element_process, display_results, get_mark_max_gflops, get_mark_min_temp
from optim_algo import retrieveMethod, basic_hill_climbing, tabu_simulated_annealing

from run_session import RunSession

from mpi4py import MPI
import mpi4py

# MPI variables for the communications inter-processes
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

parameters = {
    "is_display" : False,
    "process" : Me,
    "folder_path" : "iso3dfd-st7"
}

#####################################################################
############################ INIT PROBLEM ###########################
#####################################################################

run_session = RunSession(parameters)

#####################################################################
############################### PROBLEM #############################
#####################################################################

#S_best, E_best, k = tabu_simulated_annealing(f_cost_time, domain, temperature = 100, temp_decrease_factor= 0.95, tabu_length = 20, k_max = 400)
S_best, E_best, k = run_session()

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