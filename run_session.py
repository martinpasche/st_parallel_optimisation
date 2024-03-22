from optim_algo import retrieveMethod
from utils import Element, Domain, cmdLineParsing, cost_function_benchmark

DefaultIterMax  = 100
DefaultMethod   = "hc"
DefaultSeed     = 100
DefaultT0       = 100.0
DefaultLa       = 0.9
DefaultLtl      = 10
DefaultCostFunc = "time"
DefaultVNSHCMax = 50
DefaultVNSNeighborDistance = 9

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
    parser.add_argument("--vnsndist", help="VNS distance to  neighbors", default=DefaultVNSNeighborDistance, type=np.int32)
    
    args = parser.parse_args()

    if args.i <= 0:
        if Me == 0:
        print("Error: maximum nb of iterations must be an integer greater than 0!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)
    
    if args.m not in ("hc", "sa", "tsa", "vns", "random"):
        if Me == 0:
        print("Error: local method must be in: [hc, sa, tsa, vns, random]!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)
        
    if args.seed < 0:
        if Me == 0:
        print("Error: seed increase must be >= 0!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)
        
    if args.t0 <= 0:
        if Me == 0:
        print("Error: initial temperature must be > 0!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(1)

    if args.la <= 0:
        if Me == 0:
        print("Error: lambda parameter must be > 0!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)

    if args.ltl <= 0:
        if Me == 0:
        print("Error: length of Tabu list must be > 0!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)
    
    if args.cost not in ("time", "gflops", "mpoints"):
        if Me == 0:
        print("Error: cost function must be in [time, gflops, mpoints]",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)

    if args.vnshcmax <= 0:
        if Me == 0:
        print("Error: the number of iterations of HC must be > 0!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)

    if args.vnsndist <= 0:
        if Me == 0:
        print("Error: distance to neighbors in VNS must be > 0!",file=sys.stderr)
        sys.exit(0)
        else: 
        sys.exit(0)
    
    return args.i, args.m, args.seed, args.t0, args.la, args.ltl, args.cost, args.vnshcmax, args.vnsndist

class RunSession :
    def __init__(self, parameters : dict, domain : Domain) :
        self.process = parameters["process"]
        self.is_display = parameters["is_display"]
        self.folder_path = parameters["folder_path"]
        self.parameters = parameters
        self.domain = domain
        self.k_max, self.method_str, self.SeedInc, self.temperature, self.temp_decrease_factor, self.tabu_length, self.cost_func_str, self.vnshcmax, self.vnsndist = cmdLineParsing(self.process)

        self.setMethod(self.method_str)
        self.setCostFunction(self.cost_func_str)
        self.setStartingPoint(domain)

    def __call__(self) :
        self.display()
        return self.method(self.cost_function, self.starting_point, self.domain, self.temperature, self.temp_decrease_factor, self.tabu_length, self.k_max, self.vnshcmax, self.vnsndist)

    def display(self) :
        print(f"Currently running on {self.method_str}") 

    def setStartingPoint(self, domain) :
        self.starting_point = domain.get_random_element()

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