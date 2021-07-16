# BayesianOpt4dftu #

This code determines the Hubbard U parameters in DFT+U via Bayesian Optimization approach.

## Requirements ##

1. Python 3.6+
2. NumPy
3. Pandas
4. ASE (https://wiki.fysik.dtu.dk/ase/)
5. pymatgen (https://pymatgen.org/)
6. bayesian-optimization https://github.com/fmfn/BayesianOptimization
7. Vienna Ab initio Simulation Package (VASP) https://www.vasp.at/

## Set up the input file (input.json) before running the code 

The input file contains these parts:
- structure_info : Includes geometry information (such as lattice parameter, lattice vectors, atomic position, etc) of the 
target materials.
- general_flags: Includes general flags required in the VASP calculation.
- scf: Flags required particularly in SCF calculation.
- band: Flags required particularly in band structure calculation.
- pbe: Flags required when using PBE as exchange-correlation functional.
- hse: Flags required when using HSE06 as exchange-correlation functional.
The flags can be added or removed. More flag keys can be found in the ASE VASP calculator.

## Installation

* `pip install BayesOpt4dftu`

## Usage
Before running, change the environment variables VASP_RUN_COMMAND, OUTFILENAME, and VASP_PP_PATH.

* `cd example/`
* `python ./example.py`

## Citation
Please cite the following work if you use this code.

[1] M. Yu, S. Yang, C. Wu, N. Marom, Machine learning the Hubbard U parameter in DFT+ U using Bayesian optimization, npj Computational Materials, 6(1):1–6, 2020.

