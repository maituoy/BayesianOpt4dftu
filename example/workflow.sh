#!/bin/bash
#SBATCH -J fw  # Job name
#SBATCH -n 56 # Number of total cores
#SBATCH -N 1  # Number of nodes
#SBATCH --mem-per-cpu=1000# Memory pool for all cores in MB (see also --mem-per-cpu)
#SBATCH -o stdout # File to which STDOUT will be written %j is the job #
#SBATCH -p debug
#SBATCH -A marom
#SBATCH -x f007

python example.py

