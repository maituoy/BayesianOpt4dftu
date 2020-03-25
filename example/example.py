from BayesOpt4dftu.core import *
import os
import argparse

# Command to run VASP executable.
VASP_RUN_COMMAND = 'mpirun -np 56 vasp_ncl'
# Define the name for output file.
OUTFILENAME = 'vasp.out'
# Define the path direct to the VASP pesudopotential.
VASP_PP_PATH = '/home/maituoy/pp_vasp/'


def parse_argument():
    """
    kappa: The parameter to control exploration and exploitation.
           exploitation 0 <-- kappa --> 10 exploration

    alpha1: Weight coefficient of band gap in the objective function.

    alpha2: Weight coefficient of delta band in the objective function.
    
    threshold: Convergence threshold of Bayesian optimization process.
    """
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--which_u', dest='which_u', type=tuple, default=(1,0))
    parser.add_argument('--br', dest='br', type=int, default=4)
    parser.add_argument('--kappa', dest='kappa', type=float, default=2.5)
    parser.add_argument('--alpha1', dest='alpha1', type=float, default=1)
    parser.add_argument('--alpha2', dest='alpha2', type=float, default=1)
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.001)
    parser.add_argument('--urange', dest='urange', type=tuple, default=(-10,10))
    parser.add_argument('--import_kpath', dest='import_kpath', type=bool, default=False)

    return parser.parse_args()

def main():
    args = parse_argument()
    k = args.kappa
    a1 = args.alpha1
    a2 = args.alpha2
    which_u = tuple(int(x) for x in args.which_u)
    urange = tuple(float(x) for x in args.urange)
    br = args.br
    import_kpath = args.import_kpath

    os.environ['VASP_PP_PATH'] = VASP_PP_PATH

    calculate(command=VASP_RUN_COMMAND, outfilename=OUTFILENAME, method='hse', import_kpath = import_kpath)
    
    if os.path.exists('u.txt'):
        os.remove('u.txt')

    obj = 0 
    threshold = args.threshold
    for i in range(50):
        calculate(command=VASP_RUN_COMMAND, outfilename=OUTFILENAME, method='dftu', import_kpath = import_kpath)
        db = delta_band(bandrange=br, path='./')
        db.deltaBand()

        bayesianOpt = bayesOpt_DFTU(path='./', kappa=k, alpha_1=a1, alpha_2=a2 )
        obj_next = bayesianOpt.bo(which_u, urange)
        print(obj_next)
        if abs(obj_next - obj) <= threshold:
            print("Optimization has been finished!")
            break
        obj = obj_next        

if __name__ == "__main__":
    main()
    

