import numpy as np
from utils import cost_function_benchmark
# Benchmark Functions
def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    return np.sum(np.asarray([np.sum(x[:i+1])**2 for i in range(len(x))]))

def F4(x):
    return np.max(np.abs(x))

def F5(x):
    return np.sum(100*(x[1:]-x[:-1]*2)**2 + (x[:-1]-1)**2)

def F6(x):
    return np.sum((np.abs(x+0.5))**2)

def F7(x):
    return np.sum((np.arange(1, len(x)+1))*(x*4)) + np.random.uniform()

def F8(x):
    return np.sum(-x*np.sin(np.sqrt(np.abs(x))))

def F9(x):
    dim = len(x)
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10*dim)

def F10(x):
    dim = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) - np.exp(np.sum(np.cos(2*np.pi*x))/dim) + 20 + np.e

def F11(x):
    dim = len(x)
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, dim+1)))) + 1

def F12(x):
    dim = len(x)
    return (np.pi/dim) * (10*np.sin(np.pi*(1+(x[0]+1)/4))*2 + np.sum(((x[:-1]+1)/4)**2 * (1+10*np.sin(np.pi*(1+(x[1:]+1)/4))*2)) + ((x[-1]+1)/4)**2) + np.sum(Ufun(x, 10, 100, 4))

def F13(x):
    return 0.1 * (np.sin(3*np.pi*x[0])*2 + np.sum((x[:-1]-1)**2 * (1+np.sin(3*np.pi*x[1:])**2)) + (x[-1]-1)**2 * (1+np.sin(2*np.pi*x[-1])**2)) + np.sum(Ufun(x, 5, 100, 4))

def F14(x):
    return cost_function_benchmark(element=x,folder_path = 'iso3dfd-st7',process =0,is_display = True)
# Add definitions for F14 to F23 and Ufun as per the MATLAB code

def Ufun(x, a, k, m):
    return k * ((x-a)*m) * (x > a) + k * ((-x-a)*m) * (x < (-a))

# BenchmarkFunctions dispatcher
def BenchmarkFunctions(F):
    D = 30
    functions = {
        'F1': (lambda: (-100, 100, D, F1)),
        'F2': (lambda: (-10, 10, D, F2)),
        'F3': (lambda: (-100, 100, D, F3)),
        'F4': (lambda: (-100, 100, D, F4)),
        'F5': (lambda: (-30, 30, D, F5)),
        'F6': (lambda: (-100, 100, D, F6)),
        'F7': (lambda: (-1.28, 1.28, D, F7)),
        'F8': (lambda: (-500, 500, D, F8)),
        'F9': (lambda: (-5.12, 5.12, D, F9)),
        'F10': (lambda: (-32, 32, D, F10)),
        'F11': (lambda: (-600, 600, D, F11)),
        'F12': (lambda: (-50, 50, D, F12)),
        'F13': (lambda: (-50, 50, D, F13)),
        'F14': (lambda: ([256,256,256,1,100,16,16,16], [256,256,256,65,100,256,256,256], 8, F14))
    }
    lb, ub, dim, fobj = functions[F]()
    return lb, ub, dim, fobj