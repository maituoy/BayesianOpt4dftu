#!/bin/bash
module load cuda/9.0
module load intel/18.0.0.128
module load impi/2018
module load vasp/5.4.4_impi_cuda

mpirun -n 56 vasp_ncl > vasp.out
