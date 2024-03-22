from optim_algo import retrieveMethod
from utils import Element, Domain, cmdLineParsing, cost_function_benchmark
import argparse
import sys
import numpy as np


Olevel_list = ["-O3", "-Ofast", "-O2", "-O1"]
simd_list = ["sse", "avx", "avx2", "avx4", "avx8", "avx16", "avx32", "avx64", "avx128", "avx256", "avx512"]#, "avx2", "avx512", "sse"]
problem_size_list1 = [256]
problem_size_list2 = [256]
problem_size_list3 = [256]
cache1_list = list(range(16, 257, 16))
cache2_list = list(range(1, 65, 1))
cache3_list = list(range(1, 65, 1))
iterations_list = [100]
threads_list = list(range(1, 65))

DefaultIterMax  = 100
DefaultMethod   = "hc"
DefaultSeed     = 100
DefaultT0       = 100.0
DefaultLa       = 0.9
DefaultLtl      = 10
DefaultCostFunc = "time"
DefaultVNSHCMax = 50
DefaultVNSNeighborDistance = 9
DefaultDOlevel = "-O3,-Ofast"
DefaultDsimd = "avx,avx2,avx16,avx512,sse"
DefaultThreads = "1:64"

def cmdLineParsing(Me):
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="maximum nb of iterations per method", default=DefaultIterMax, type=np.int32)
    parser.add_argument("--m", help="local method (hc, sa, tsa, vns, random)", default=DefaultMethod)
    parser.add_argument("--seed", help="automatic increase of the seed (of random gen.) on each process", default=DefaultSeed, type=np.int32)
    parser.add_argument("--t0", help="initial temperature t0", default=DefaultT0, type=np.float64)
    parser.add_argument("--la", help="lambda parameter of temperature law", default=DefaultLa, type=np.float64)
    parser.add_argument("--ltl", help="length of Tabu list", default=DefaultLtl, type=np.int32)
    parser.add_argument("--cost", help="cost criterium (time, gflops, mpoints)", default=DefaultCostFunc)
    parser.add_argument("--vnshcmax", help="VNS maximum HC iterations in local search", default=DefaultVNSHCMax, type=np.int32)
    parser.add_argument("--vnsndist", help="VNS distance to neighbors", default=DefaultVNSNeighborDistance, type=np.int32)
    parser.add_argument("--dOlevel", help="set domain possible Olevels (-O3,-Ofast) -> separate different values with commas", default=DefaultDOlevel)
    parser.add_argument("--dsimd", help="set domain possible simd (avx,avx2,avx16,avx512,sse) -> separate different values with commas", default=DefaultDsimd)
    parser.add_argument("--dthreads", help="set domain possible range of threads number (1:64)", default=DefaultThreads)
    parser.add_argument("--start", help="set starting point of paths", default="")
    
    args = parser.parse_args()

    if args.i <= 0:
        if Me == 0:
            print("Error: maximum nb of iterations must be an integer greater than 0!",file=sys.stderr)
        sys.exit(0)
    
    if args.m not in ("hc", "sa", "tsa", "vns", "random"):
        if Me == 0:
            print("Error: local method must be in: [hc, sa, tsa, vns, random]!",file=sys.stderr)
        sys.exit(0)
        
    if args.seed < 0:
        if Me == 0:
            print("Error: seed increase must be >= 0!",file=sys.stderr)
        sys.exit(0)
        
    if args.t0 <= 0:
        if Me == 0:
            print("Error: initial temperature must be > 0!",file=sys.stderr)
        sys.exit(0)

    if args.la <= 0:
        if Me == 0:
            print("Error: lambda parameter must be > 0!",file=sys.stderr)
        sys.exit(0)

    if args.ltl <= 0:
        if Me == 0:
            print("Error: length of Tabu list must be > 0!",file=sys.stderr)
        sys.exit(0)
    
    if args.cost not in ("time", "gflops", "mpoints"):
        if Me == 0:
            print("Error: cost function must be in [time, gflops, mpoints]",file=sys.stderr)
        sys.exit(0)

    if args.vnshcmax <= 0:
        if Me == 0:
            print("Error: the number of iterations of HC must be > 0!",file=sys.stderr)
        sys.exit(0)

    if args.vnsndist <= 0:
        if Me == 0:
            print("Error: distance to neighbors in VNS must be > 0!",file=sys.stderr)
        sys.exit(0)

    DefaultDOlevel = "-O3,-Ofast"
    DefaultDsimd = "avx,avx512,sse"
    DefaultThreads = "1:64"

    possibleOlevels = args.dOlevel.strip("[]() ").split(",")
    if any([possibleOlevel not in Olevel_list for possibleOlevel in possibleOlevels]) :
        if Me == 0:
            print("Error: Olevel must be in [-O3, -Ofast, -O2, -O1]",file=sys.stderr)
        sys.exit(0)

    possiblesimds = args.dsimd.strip("[]() ").split(",")
    if any([possiblesimd not in simd_list for possiblesimd in possiblesimds]) :
        if Me == 0:
            print("Error: simd must be in [sse, avx, avx2, avx4, avx8, avx16, avx32, avx64, avx128, avx256, avx512]",file=sys.stderr)
        sys.exit(0)

    possibleThreads = map(lambda x : int(x), args.dthread.strip("[]() ").split(":"))
    if len(possibleThreads) != 2 or any([possibleThread not in threads_list for possibleThread in possibleThreads]) :
        if Me == 0:
            print("Error: threads range limits must be in [1,64] (inclusive) (example 1:64 or 32:64)",file=sys.stderr)
        sys.exit(0)
    possibleThreads = range(possibleThreads[0], possibleThreads[1]+1)

    Olevel_list = ["-O3", "-Ofast", "-O2", "-O1"]
    simd_list = ["sse", "avx", "avx2", "avx4", "avx8", "avx16", "avx32", "avx64", "avx128", "avx256", "avx512"]#, "avx2", "avx512", "sse"]
    problem_size_list1 = [256]
    problem_size_list2 = [256]
    problem_size_list3 = [256]
    cache1_list = list(range(16, 257, 16))
    cache2_list = list(range(1, 65, 1))
    cache3_list = list(range(1, 65, 1))
    iterations_list = [100]
    threads_list = list(range(1, 65))

    start_array = args.start.split(",")
    if len(start_array) == 10 :
        for i in range(2,10) :
            start_array[i] = int(start_array[i])
    if args.start != "" and (len(start_array) != 10 or
        start_array[0] not in possibleOlevels or
        start_array[1] not in possiblesimds or
        start_array[2] in problem_size_list1 or start_array[3] in problem_size_list2 or start_array[4] in problem_size_list3 or
        start_array[5] in threads_list or
        start_array[6] in iterations_list or
        start_array[7] in cache1_list or start_array[8] in cache2_list or start_array[9] in cache3_list):
        if Me == 0:
            print("Error: starting point invalid (example -O3,avx512,256,256,256,32,100,128,7,7)",file=sys.stderr)
        sys.exit(0)
    
    return args.i, args.m, args.seed, args.t0, args.la, args.ltl, args.cost, args.vnshcmax, args.vnsndist, (possibleOlevels, possiblesimds, possibleThreads), start_array

class RunSession :
    def __init__(self, parameters : dict, domain : Domain) :
        self.process = parameters["process"]
        self.is_display = parameters["is_display"]
        self.folder_path = parameters["folder_path"]
        self.parameters = parameters
        self.k_max, self.method_str, self.SeedInc, self.temperature, self.temp_decrease_factor, self.tabu_length, self.cost_func_str, self.vnshcmax, self.vnsndist, possibleDomains, start = cmdLineParsing(self.process)
        self.domain = Domain(possibleDomains[0], possibleDomains[1], problem_size_list1, problem_size_list2, problem_size_list3, cache1_list, cache2_list, cache3_list, iterations_list, possibleDomains[2])

        self.setMethod(self.method_str)
        self.setCostFunction(self.cost_func_str)
        self.setStartingPoint(domain, start)
        
        if parameters.Me == 0:
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

    def __call__(self) :
        self.display()
        return self.method(self.cost_function, self.starting_point, self.domain, self.temperature, self.temp_decrease_factor, self.tabu_length, self.k_max, self.vnshcmax, self.vnsndist)

    def display(self) :
        print(f"Currently running on {self.method_str}") 

    def setStartingPoint(self, domain, start) :
        if len(start) == 1 :
            self.starting_point = domain.get_random_element()
        else :
            self.starting_point = Element(start[0], start[1], start[2], start[3], start[4], start[5], start[7], start[8], start[9], start[6])

    def setMethod(self, method_str) :
        self.method = retrieveMethod(method_str)

    def setCostFunction(self, cost_func_str) :
        f_costs = [self.f_cost_time, self.f_cost_gflops, self.f_cost_mpoints]
        cost_mapper = {"time" : 0, "gflops" : 1, "mpoints" : 2}
        self.cost_function = f_costs[cost_mapper[cost_func_str]]
    
    #defining cost function for time, mpoints and gflops
    def f_cost_time(self, S : Element) -> float:
        return cost_function_benchmark(S, **self.parameters).time
        
    def f_cost_mpoints(self, S : Element) -> float:
        return -cost_function_benchmark(S, **self.parameters).mpoints

    def f_cost_gflops(self, S : Element) -> float:
        return -cost_function_benchmark(S, **self.parameters).flops