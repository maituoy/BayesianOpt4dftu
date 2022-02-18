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
8. Vaspvis (https://github.com/DerekDardzinski/vaspvis)

## Set up the input file (input.json) before running the code 

The input file contains these parts:
- structure_info : Includes geometry information (such as lattice parameter, lattice vectors, atomic position, etc) of the 
target materials.
  #### Example based on InAs:
    1. Lattice_param and cell: define the 2nd to 5th rows in your POSCAR.
      
      "lattice_param": 6.0584,
        "cell": [
            [
                0.0,
                0.5,
                0.5
            ],
            [
                0.5,
                0.0,
                0.5
            ],
            [
                0.5,
                0.5,
                0.0
            ]
        ]
    2. Atoms: Define the atomic positions of each atom in your system and the initial magnetic moment if there is any.
      
      With SOC:
      "atoms": [
            [
                "In",
                [
                    0,
                    0,
                    0
                ],
                [
                    0,
                    0,
                    1e-06
                ]
            ],
            [
                "As",
                [
                    0.75,
                    0.75,
                    0.75
                ],
                [
                    0,
                    0,
                    1e-06
                ]
            ]
        ]
        
      Without SOC:
        "atoms": [
            [
                "In",
                [
                    0,
                    0,
                    0
                ],
                 1e-06
            ],
            [
                "As",
                [
                    0.75,
                    0.75,
                    0.75
                ],
                 1e-06
            ]
        ]
     So in this case, there are two atoms in the primitive cell which are located at the position `(0,0,0)` and `(0.75, 0.75, 0.75)`. The second term under each atom defines the initial magnetic moment. If the spin-orbit coupling is not included in your calculation, it is just an integer while otherwise it is a (3,) array of each element defines the initial moment of corresponding direction. If the initial moment is 0, it has to be set to a small number to avoid confliction in the ASE.
- general_flags: Includes general flags required in the VASP calculation.
- scf: Flags required particularly in SCF calculation.
- band: Flags required particularly in band structure calculation.
- pbe: Flags required when using PBE as exchange-correlation functional.
- hse: Flags required when using HSE06 as exchange-correlation functional.
The flags can be added or removed. More flag keys can be found in the ASE VASP calculator.

## Installation

* `pip install BayesOpt4dftu`

## Usage
To run the examples in the `example` folder, you need to modify the environment settings in the `example.py` of the selected calculation based on the specs of your system and VASP binary.

I will use `/example/2d` as an example:
  
#### 1. Setting environments
  
  Set the running command for VASP executable
  
      VASP_RUN_COMMAND = 'srun -n 54 vasp_ncl'
      
  Define the VASP output file name.
      
      OUTFILENAME = 'vasp.out'
      
  Define the path direct to the VASP pesudoopotential. (P.S. It should be the directory containing the `potpaw_PBE` folder)
      
      VASP_PP_PATH = '/PATH/TO/THE/PESUDOPOTENTIAL/'
      
#### 2. Arguments options

  `--which_u` defines which element you would like to optimize the U for. For a unary substance, it has to be `(1,)`. For compounds with over 2 elements, you can set each element to 0 or 1 to switch on/off the optimization for that element. For InAs, when optimizing for both In and As, it will be `(1,1)`.
  
  `--br` defines band range you would like to include in your Δband. It is a tuple of two integers, which define the number of valence bands and conduction bands from the Fermi level.
  
  `--kappa` controls the exploration and exploitation when acquisition function sample the next points. Exploitation 0 <-- kappa --> 10 Exploration
  
  `--alpha1` and `alpha2` are the weight coefficients of Δgap and Δband.
  
  `--threshold` defines at what accuracy would you stop the BO process.
  
  `--urange` defines the U parameter range for optimization, currently it's not supported to define different U ranges for different elements.
  
  `--import_kpath` provides an external listing of high symmetry k-points in case some special k coordinates are not present in the ase library.
  
  `--elements` defines the elements in your system. It is set for plotting the BO results. If it's a unary substance, it has to be (ele,).
  
  `--iteration` defines the maximum steps that BO will be performed with.

#### 3. Running the code

  After setting up all these stuff, you can simply run the calculation by
  
  `cd example/1d`
  
  `python example.py --arg1 XX --arg2 XX ...`
  
#### 4. Outputs

  Once the threshold or the maximum iterations is reached, you will get two output files
  
  `u_xx.txt` file consists of the U parameters, band gap, and the Δband at each step.
  
  `1D_xxx.png` or `2D_xxx.png` plots showing you the Gaussian process predicted mean and the acquisition function.
  
  Example of BO plots
  
  <img src="https://github.com/maituoy/BayesianOpt4dftu/blob/master/example/1d/1D_kappa_5_a1_0.5_a2_0.5.png" width="600" height="400">
  
  <img src="https://github.com/maituoy/BayesianOpt4dftu/blob/master/example/2d/2D_kappa_5_a1_0.25_a2_0.75.png" width="800" height="270">
  
  Optimal U values will be output at the end of entire process based on the interpolation from the predicted mean space. You can also pick up the Us that give you largest objective value from the u.txt file.

## Citation
Please cite the following work if you use this code.

[1] M. Yu, S. Yang, C. Wu, N. Marom, Machine learning the Hubbard U parameter in DFT+ U using Bayesian optimization, npj Computational Materials, 6(1):1–6, 2020.

