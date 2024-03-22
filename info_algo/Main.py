import numpy as np
import matplotlib.pyplot as plt
from benchmarkfunction import BenchmarkFunctions
from INFO import INFO
from utils import cost_function_benchmark


# Example main script setup
nP = 30  # Number of Population
MaxIt = 50  # Maximum number of iterations
Func_name = 'F14'
# Load details of the selected benchmark function
lb, ub, dim, fobj = BenchmarkFunctions(Func_name)


# Run the INFO algorithm
Best_fitness, BestPositions, Convergence_curve = INFO(nP, MaxIt, lb, ub, dim, fobj)

# Draw objective space
plt.figure()
plt.semilogy(Convergence_curve, color='r', linewidth=4)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.tight_layout()
plt.grid(False)
plt.box_on = True
plt.legend(['INFO'])
plt.show()