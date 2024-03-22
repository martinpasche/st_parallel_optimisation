import os
import subprocess
import random
from itertools import product
from typing import List
from mpi4py import MPI
import mpi4py
import random
import re
import argparse
import numpy as np
import time as tm

class Element:
    
    def __init__(self, Olevel, simd, problem_size1, problem_size2, problem_size3, threads, cache1, cache2, cache3, iter = 100, code = ()):
        
        self.simd = simd
        self.Olevel = self.correctly_define_Olevel(Olevel)
        self.problem_size1 = problem_size1
        self.problem_size2 = problem_size2
        self.problem_size3 = problem_size3
        self.threads = threads
        self.cache1 = cache1
        self.cache2 = cache2
        self.cache3 = cache3
        self.iter = iter
        self.code = code
        
    def __str__(self):
        text = "[" + str(self.Olevel).ljust(7) + str(self.simd).ljust(7) \
            + str(self.problem_size1).ljust(4) + str(self.problem_size2).ljust(4) + str(self.problem_size3).ljust(4) \
            + str(self.threads).ljust(3) \
            + str(self.iter).ljust(4) \
            + str(self.cache1).ljust(4) + str(self.cache2).ljust(4) + str(self.cache3).ljust(4) + "]"
        return text
    
    def __repr__(self) -> str:
        text = "[" + str(self.Olevel).ljust(7) + str(self.simd).ljust(7) \
            + str(self.problem_size1).ljust(4) + str(self.problem_size2).ljust(4) + str(self.problem_size3).ljust(4) \
            + str(self.threads).ljust(3) \
            + str(self.iter).ljust(4) \
            + str(self.cache1).ljust(4) + str(self.cache2).ljust(4) + str(self.cache3).ljust(4) + "]"
        return text
    
    
        
    def correctly_define_Olevel(self, Olevel):
        if type(Olevel) != str:
            raise Exception("Olevel of different type")
        
        Olevel = Olevel.strip()
        
        if Olevel[0] != "-":
            return "-" + Olevel
        else:
            return Olevel
        


class BenchMarkValue:
    
    def __init__(self, time, mpoints, flops, element : Element):
        self.time = float(time)
        self.mpoints = float(mpoints)
        self.flops = float(flops)
        self.element = element
        self.is_error = False
        self.check_error()
        
    def check_error (self):
        if self.time == -1 or self.flops == -1:
            self.is_error = True
    
    def __str__(self):
        return f"time: " + str(f"{self.time:.2f}").ljust(5) + " s  |  throughput: " + str(f"{self.mpoints:.2f}").ljust(5) + " Mpoints/s  |  flops: " + str(f"{self.flops:.2f}").ljust(5) + " Gigaflops"




class Domain:
    
    def __init__(self, Olevel_list, simd_list, problem_size1_list, problem_size2_list, problem_size3_list, cache1_list, cache2_list, cache3_list, threads_list, iterations_list):
        self.__variables = [
            Olevel_list,
            simd_list,
            iterations_list,
            threads_list,
            problem_size1_list,
            problem_size2_list,
            problem_size3_list,
            cache1_list,
            cache2_list,
            cache3_list
        ]
        
    def get_random_element (self) -> Element:
        indexes = []
        for i in range(len(self.__variables)):
            index = random.randint(0, len(self.__variables[i]) - 1)
            indexes.append(index)
        return self.get_element(indexes)
            
    
    def get_element (self, array_index):
        try:
            return Element(
                Olevel = self.__variables[0][array_index[0]],
                simd = self.__variables[1][array_index[1]],
                iter = self.__variables[2][array_index[2]],
                threads= self.__variables[3][array_index[3]],
                problem_size1 = self.__variables[4][array_index[4]],
                problem_size2 = self.__variables[5][array_index[5]],
                problem_size3 = self.__variables[6][array_index[6]],
                cache1 = self.__variables[7][array_index[7]],
                cache2 = self.__variables[8][array_index[8]],
                cache3 = self.__variables[9][array_index[9]],
                code = array_index
            )
        except Exception as error:
            # print(error)
            return None

    def get_small_neighborhood(self, element) :
        """ 
        index 0: Olevel 
        index 1: simd
        index 2: iter
        index 3: threads
        index 4: problem_size1
        index 5: problem_size2
        index 6: problem_size3
        index 7: cache1
        index 8: cache2
        index 9: cache3
        """
        
        blocked_index = (2, 4, 5, 6)

        elements_code = []

        for i in range(len(self.__variables)) :
            if i in blocked_index :
                continue

            for incr in [-1, +1] :
                element_code_copy = element.code.copy()
                element_code_copy[i] += incr
                if element_code_copy[i] >= 0 and element_code_copy[i] < len(self.__variables[i]) :
                    elements_code.append(element_code_copy)
        
        #Remove duplicates
        elements_code = map(lambda elem_code : list(elem_code), list(set(map(lambda elem_code : tuple(elem_code), elements_code))))
        elements = list(map(lambda elem_code : self.get_element(elem_code), elements_code))
        return elements
        
    
    def get_neighborhood(self, element):
        """ 
        index 0: Olevel 
        index 1: simd
        index 2: iter
        index 3: threads
        index 4: problem_size1
        index 5: problem_size2
        index 6: problem_size3
        index 7: cache1
        index 8: cache2
        index 9: cache3
        """

        element_code = element.code
        possible_codes = []
        blocked_index = (2, 4, 5, 6)
        
        for i in range(10):
            if i not in blocked_index:
                possible_codes.append([element_code[i] - 1, element_code[i], element_code[i] + 1])
            else:
                possible_codes.append([element_code[i]])
                
        combinations = list(product(*possible_codes))
        
        elements = list( map( lambda combination: self.get_element(combination), combinations ))
        elements_filtered = list(filter( lambda x: x != None and x.code == element.code, elements))
        
        #print(elements_filtered)
        return elements_filtered
    
    def VNS_neighbor(self, element : Element, previous_neighbors : List[Element] = [], k=1):
     
        neighbors = self.get_small_neighborhood(element)
        previous_neighbors.append(element)
        
        #Choose a neighbor without going back to a previous neighbor
        neighbor = previous_neighbors[0]
        size = len(neighbors)
        repetitions = 0
        while neighbor in previous_neighbors:
            neighbor = random.choice(neighbors)
            repetitions += 1
            if repetitions>2*size:
                raise ValueError("Solution not found in list")
                    
        if k==1:
            return neighbor
        #Recursion to find a (k-1)th neighbor of this new neighbor, which would be a kth neighbor of element, without going back
        else:
            return self.VNS_neighbor(neighbor, previous_neighbors, k-1)
        
        
        
        
def display_element_process (element : Element, process = -1, content = ""):
    print(f"Process {process}\t{element}\t{content}")       
    

def display_results (marks : List[BenchMarkValue] = []):
    print("\nDisplaying the best solution of each node\n")
    for mark in marks:
        print(mark.element, mark, " Iter:", mark.k)
    
    
def get_mark_min_temp (marks : List[BenchMarkValue] = None) -> BenchMarkValue:
    try:
        min_temp_mark = min( marks, key = lambda x: x.time )
        return min_temp_mark
    except Exception as error:
        print(error)
        return None
    
def get_mark_max_gflops (marks : List[BenchMarkValue] = None) -> BenchMarkValue:
    try:
        mark_max_gflops = max( marks, key = lambda x: x.flops )
        return mark_max_gflops
    except Exception as error:
        print(error)
        return None
    




def cmdEnterPath (path):
    try:
        os.chdir(path)
    except Exception as e:
        pass


def cmdRunModule ():
    try:
        os.system("module load intel-oneapi-compilers/2023.1.0/gcc-11.4.0")
    except Exception as error:
        pass
        

def cmdMake( element : Element = None, process = 0, is_display : bool = False):
    try:
        level_str = f"Olevel={element.Olevel}"
        simd_str = f"simd={element.simd}"
        process_str = f"process={process}"
        
        checking_path = os.path.join("bin", f"iso3dfd_dev13_cpu{element.Olevel}_{element.simd}.exe")
        is_file = os.path.isfile(checking_path)
        
        if not is_file:
            if is_display:
                display_element_process(element, process, "Making file!!!")
            cmd = f"make {simd_str} {level_str} last"

            text = subprocess.run( cmd, shell=True, stdout=subprocess.PIPE)

            #Renaming the file
            current_path = os.path.join("bin","iso3dfd_dev13_cpu_"+element.simd+".exe")
            os.rename(current_path, checking_path)
            
        else:
            if is_display:
                display_element_process(element, process, "File already exists!!!")
        
    except Exception as error:
        print(error)
        
        

SEARCH_PATTERN = r"time:\s+(\d+\.\d+)\s+sec throughput:\s+(\d+\.\d+)\s+MPoints\/s flops:\s+(\d+\.\d+)\s+GFlops"
def get_output_info (result):
    data = " ".join(result.stdout.strip().split("\n")[-3:])
    match = re.search(SEARCH_PATTERN, data)

    return (float(match.group(i)) for i in range(1,4))


def cmdRunIso (element : Element = None, process = 0, is_display : bool = False):

    #path = os.path.join("bin", f"iso3dfd_dev13_cpu_{element.simd}.exe")
    path = os.path.join("bin", f"iso3dfd_dev13_cpu{element.Olevel}_{element.simd}.exe")
    
    cmd = f"{path} {element.problem_size1} {element.problem_size2} {element.problem_size3} {element.threads} {element.iter} {element.cache1} {element.cache2} {element.cache3}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    #print(result.stdout)
    time, mpoints, flops = get_output_info(result)
        
    mark = BenchMarkValue(time, mpoints, flops, element)
    display_element_process(element, process, mark)
    return mark
    
    
    
    
def cost_function_benchmark( element : Element, folder_path="iso3dfd-st7", process = 0, is_display = True ) -> BenchMarkValue:
    
    
    cmdEnterPath(folder_path)

    #cmdMake( element = element, Olevel=element.Olevel, simd = element.simd, process=process )
    cmdMake( element, process, is_display)
    mark = cmdRunIso(element= element, process = process, is_display = is_display)
    return mark