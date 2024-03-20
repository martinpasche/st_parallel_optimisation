# Comandos para correr mpi y ssh

Para visualizar en VSC Ctrl+Shift+V


## SSH

### Conectarse entorno virtual

    ssh -l st76i_11 chome.metz.supelec.fr

### Pedir una sesion

#### Normal

    srun -N 1 --exclusive -p cpu_tp --pty bash

#### Reserva

    srun -N1 --exclusive --reservation=XXXXX --pty bash

#### MPI

Para mpi, se necesita pedir todo el core

    srun -N 1 --exclusive -p cpu_tp --pty bash

#### Check sessions

    mysrun

#### Cancel a job

    scancel <jobId>



### Send file to remote machine

    scp -r <local file> st76i_11@chome.metz.supelec.fr:




## MPI 

### Adding to SSh

    module load py-mpi4py/....


### MPI command

    mpirun -np <#nb> 
    -map-by ppr:<NbProcessParRessource>:<node|socket|core>:PE=<NbCoeursPhysiquesAccesibleParProccesus>
    -rank-by <node|socket|core>
    python3 ./main.py

Maybe one the best configurations
M is the number of machines available and Cn process a creer

    mpirun -np M*Cn -map-by ppr:1:core:PE=1 -rank-by core



## Iso file

    module load intel-oneapi-compilers/2023.1.0/gcc-11.4.0
    make Olevel=-03 simd=avx2 last

We have to remember that the module load and the make have to be run in the archive of the iso

Running the iso file

    bin/iso3dfd_dev_cpu_avx2.exe cache1 cache2 cache3 threads iter small-cache1 small-cache2 small-cache3
    
    

## Check cache misses

    valgrind --tool=cachegrind /usr/users/st76i/st76i_5/iso3dfd-st7/bin/iso3dfd_dev13_cpu_avx.exe