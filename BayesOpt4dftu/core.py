import os
import json
import bayes_opt
import subprocess
import numpy as np
import pandas as pd
import pymatgen as mg
import xml.etree.ElementTree as ET

from ase import Atoms, Atom
from ase.calculators.vasp.vasp2 import Vasp2
from ase.dft.kpoints import *

from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen import Lattice, Structure, Molecule
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from string import ascii_lowercase

def readgap(vasprun, kpoints):
    run = BSVasprun(vasprun)
    bs = run.get_band_structure(kpoints)
    if (bs.is_metal()==False):
        return bs.get_cbm()['energy']-bs.get_vbm()['energy']
    else:
        return 0


class vasp_init(object):
    def __init__(self, input_path):
        with open(input_path, 'r') as f:
            self.input_dict = json.load(f)
        self.struct_info = self.input_dict['structure_info']
        self.general_flags = self.input_dict['general_flags']
        self.atoms = None
    
    def init_atoms(self):
        lattice_param = self.struct_info['lattice_param']
        cell = np.array(self.struct_info['cell'])
        self.atoms = Atoms(cell=cell*lattice_param)
        for atom in self.struct_info['atoms']:
            self.atoms.append(Atom(atom[0], atom[1], magmom=atom[2]))
        
        return self.atoms

    def modify_poscar(self, path='./'):
        with open(path + '/POSCAR', 'r') as f:
            poscar = f.readlines()
            poscar.insert(5, ' '.join(x.symbol for x in self.atoms) + '\n')
            poscar[7] = 'Direct\n'
            f.close()

        with open(path + '/POSCAR','w') as d:
            d.writelines(poscar)
            d.close()
    
    def kpt4pbeband(self, path):
        special_kpoints = get_special_points(self.atoms.cell)
        num_kpts = self.struct_info['num_kpts']
        labels=self.struct_info['kpath']
        kptset = list()
        lbs = list()
        if labels[0] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[0]])
            lbs.append(labels[0])

        for i in range(1,len(labels)-1):
            if labels[i] in special_kpoints.keys():
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
        if labels[-1] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[-1]])
            lbs.append(labels[-1])

        kpt = Kpoints(comment='band', kpts=kptset, num_kpts = num_kpts,
                      style='Line_mode',coord_type="Reciprocal",labels=lbs)
        kpt.write_file(path+'/KPOINTS')
    
    def kpt4hseband(self, path):
        ibz = open(path+'/IBZKPT','r')
        num_kpts = self.struct_info['num_kpts']
        labels=self.struct_info['kpath']
        ibzlist = ibz.readlines()
        ibzlist[1] = str(num_kpts*(len(labels)-1) + int(ibzlist[1].split('\n')[0])) + '\n'
        special_kpoints = get_special_points(self.atoms.cell)
        for i in range(len(labels)-1):
            k_head = special_kpoints[labels[i]]
            k_tail = special_kpoints[labels[i+1]]
            increment = (k_tail-k_head)/(num_kpts-1)
            ibzlist.append(' '.join(map(str, k_head)) + ' 0 ' + labels[i] + '\n')
            for j in range(1,num_kpts-1):
                k_next = k_head + increment*j
                ibzlist.append(' '.join(map(str, k_next)) + ' 0\n')
            ibzlist.append(' '.join(map(str, k_tail)) + ' 0 ' + labels[i+1] + '\n')
        with open(path+'/KPOINTS','w') as f:
            f.writelines(ibzlist)
    
    def generate_input(self, directory, step, xc):
        flags = {}
        flags.update(self.general_flags)
        flags.update(self.input_dict[xc])
        flags.update(self.input_dict[step])
        if step == 'scf':
            calc = Vasp2(self.atoms,directory=directory,kpts=self.struct_info['kgrid_'+xc],gamma=True,**flags)
            calc.write_input(self.atoms)
            self.modify_poscar(path=directory)
        elif step == 'band':
            calc = Vasp2(self.atoms,directory=directory,gamma=True,**flags)
            calc.write_input(self.atoms)
            self.modify_poscar(path=directory)
            if xc == 'pbe':
                self.kpt4pbeband(directory)
            elif xc == 'hse':
                self.kpt4hseband(directory)
            

class delta_band(object):
    def __init__(self, bandrange=10, path='./'):
        self.br = bandrange
        self.vasprun_hse = path + 'hse/band/vasprun.xml'
        self.kpoints_hse = path + 'hse/band/KPOINTS'
        self.vasprun_dftu = path + 'dftu/band/vasprun.xml'
        self.kpoints_dftu = path + 'dftu/band/KPOINTS'
    
    def access_eigen(self, vasprun,spin):
        eigenval = vasprun.eigenvalues
        key = eigenval.keys()
        key_set = []
        for k in key:
            key_set.append(k)

        if spin==1:
            return eigenval[key_set[0]]
        else:
            return eigenval[key_set[1]]

    def get_vbm(self, v, e_fermi):
        min_tmp = float("-inf")
        for i,j in zip(*np.where( v <= e_fermi)):
            if v[i,j] > min_tmp:
                min_tmp = float(v[i,j])
        return min_tmp
    
    def get_cbm(self, v, e_fermi):
        max_tmp = float("inf")
        for i,j in zip(*np.where( v >= e_fermi)):
            if v[i,j] < max_tmp:
                max_tmp = float(v[i,j])
        return max_tmp
    
    def readInfo(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        ispin = int(root.findall('./parameters/separator/.[@name="electronic"]/separator/.[@name="electronic spin"]/i/.[@name="ISPIN"]')[0].text)
        nbands = int(root.findall('./parameters/separator/.[@name="electronic"]/i/.[@name="NBANDS"]')[0].text)
        nkpts = len(root.findall('./kpoints/varray/.[@name="kpointlist"]/v'))

        return ispin, nbands, nkpts
    
    def deltaBand(self):
        ispin_hse, nbands_hse, nkpts_hse = self.readInfo(self.vasprun_hse)
        ispin_dftu, nbands_dftu, nkpts_dftu = self.readInfo(self.vasprun_dftu)

        if nbands_hse != nbands_dftu:
            return print('Error: The band number of HSE and GGA+U are not match!')

        kpoints = [line for line in open(self.kpoints_hse) if line.strip()]
        kpts_diff = 0
        for ii, line in enumerate(kpoints[3:]):
            if line.split()[3] != '0':
                kpts_diff += 1

        if nkpts_hse - kpts_diff != nkpts_dftu:
            return print('Error: The kpoints number of HSE and GGA+U are not match!')
        
        run_hse = BSVasprun(self.vasprun_hse)
        bs_hse = run_hse.get_band_structure(self.kpoints_hse)

        run_dftu = BSVasprun(self.vasprun_dftu)
        bs_dftu = run_dftu.get_band_structure(self.kpoints_dftu)

        v_hse = []
        v_dftu = []
        v = {}

        if ispin_hse == 1 and ispin_dftu == 1:
            v_hse.append(self.access_eigen(run_hse,1)[kpts_diff:,:,0])
            v_dftu.append(self.access_eigen(run_dftu,1)[:,:,0])


        elif ispin_hse == 2 and ispin_dftu == 2:
            v_hse.append(self.access_eigen(run_hse,1)[kpts_diff:,:,0])
            v_hse.append(self.access_eigen(run_hse,-1)[kpts_diff:,:,0])

            v_dftu.append(self.access_eigen(run_dftu,1)[:,:,0])
            v_dftu.append(self.access_eigen(run_dftu,-1)[:,:,0])

        else:
            return print('Error: The spin number of HSE and GGA+U are not match!')
        
        v['hse'] = np.array(v_hse)
        v['dftu'] = np.array(v_dftu)

        edge = {}
        loc = {}
        efermi = {}
        efermi['hse'] = run_hse.efermi
        efermi['dftu'] = run_dftu.efermi

        for m in 'hse','dftu':
            spin = {}
            i = {}
            for s in range(ispin_hse):
                vbm = self.get_vbm(v[m][s],efermi[m])
                cbm = self.get_cbm(v[m][s],efermi[m])
                vbm_loc = max(np.where(v[m][s] == vbm)[1])
                cbm_loc = min(np.where(v[m][s] == cbm)[1])
                spin[s] = [vbm,cbm]
                i[s] = [vbm_loc,cbm_loc]
            edge[m] = spin
            loc[m] = i
        shifted_hse = np.concatenate(((v['hse'][0] - edge['hse'][0][0])[:,loc['hse'][0][0]-self.br+1:loc['hse'][0][0]+1],
                                  (v['hse'][0] - edge['hse'][0][1])[:,loc['hse'][0][1]:loc['hse'][0][1]+self.br]),
                                   axis = 1)

        if ispin_hse == 2:
            shifted_hse = np.concatenate((shifted_hse,
                                        (v['hse'][1] - edge['hse'][0][0])[:,loc['hse'][1][0]-self.br+1:loc['hse'][1][0]+1],
                                        (v['hse'][1] - edge['hse'][0][1])[:,loc['hse'][1][1]:loc['hse'][1][1]+self.br]),
                                        axis = 1)
        if ispin_dftu == 1:
            continuous = (loc['dftu'][0][1] - loc['dftu'][0][0]) == 1
        elif ispin_dftu == 2:
            continuous = (loc['dftu'][0][1] - loc['dftu'][0][0]) == 1 & (loc['dftu'][1][1] - loc['dftu'][1][0]) == 1
        else:
            return print('Error: Check your ISPIN for GGA+U')

        if bs_dftu.is_metal() == False or continuous == True:

            shifted_dftu = np.concatenate(((v['dftu'][0] - edge['dftu'][0][0])[:,loc['dftu'][0][0]-self.br+1:loc['dftu'][0][0]+1],
                                        (v['dftu'][0] - edge['dftu'][0][1])[:,loc['dftu'][0][1]:loc['dftu'][0][1]+self.br]),
                                            axis = 1)
            if ispin_dftu == 2:
                shifted_dftu = np.concatenate((shifted_dftu,
                                            (v['dftu'][1] - edge['dftu'][0][0])[:,loc['dftu'][1][0]-self.br+1:loc['dftu'][1][0]+1],
                                            (v['dftu'][1] - edge['dftu'][0][1])[:,loc['dftu'][1][1]:loc['dftu'][1][1]+self.br]),
                                            axis = 1)
        else:
            shifted_dftu = (v['dftu'][0] - edge['dftu'][0][0])[:,loc['dftu'][0][0]-self.br+1:loc['dftu'][0][0]+1+self.br]
            if ispin_dftu == 2:
                shifted_dftu = np.concatenate((shifted_dftu,
                                            (v['dftu'][1] - edge['dftu'][0][0])[:,loc['dftu'][1][0]-self.br+1:loc['dftu'][1][0]+1+self.br]),
                                            axis = 1)
    
        n = shifted_hse.shape[0] * shifted_hse.shape[1]
        delta_band = sum((1/n)*sum((shifted_hse - shifted_dftu)**2))**(1/2)

        if bs_dftu.is_metal()==False:
            gap = edge['dftu'][0][1] - edge['dftu'][0][0]
        else:
            gap = 0

        incar = Incar.from_file('./dftu/band/INCAR')
        u = incar['LDAUU']
        u.append(gap)
        u.append(delta_band)
        output = ' '.join(str(x) for x in u) 

        with open('u.txt','a+') as f:
            f.write(output + '\n')
            f.close

        return delta_band

class bayesOpt_DFTU(object):
    def __init__(self, path, kappa=2.5, alpha_1=1, alpha_2=1):
        self.input = path + 'u.txt'
        self.gap = readgap(path + '/hse/band/vasprun.xml', path + '/hse/band/KPOINTS')
        self.kappa = kappa
        self.a1 = alpha_1
        self.a2 = alpha_2
    
    def loss(self, y, y_hat, delta_band, alpha_1, alpha_2):
        return -alpha_1 * (y - y_hat) ** 2 - alpha_2 * delta_band ** 2
    
    def bo(self, num_variables):
        data = pd.read_csv(self.input, header=None, delimiter="\s",engine ='python')
        num_rows, d = data.shape
        variables_string = ascii_lowercase[:num_variables]
        pbounds = {}
        if num_variables == 1:
            pbounds[variables_string[0]] = (0,10)
        elif num_variables == 2:
            for variable in variables_string:
                pbounds[variable] = (0, 10)
        
        utility = UtilityFunction(kind="ucb", kappa=self.kappa, xi=0.0)
        optimizer = BayesianOptimization(
                                        f=None,
                                        pbounds=pbounds,
                                        verbose=2,
                                        random_state=1,
                                       )
        for i in range(num_rows):
            values = list(data.iloc[i])[:-2]
            params = {}
            for (value, variable) in zip(values, variables_string):
                params[variable] = value
            target = self.loss(self.gap, list(data.iloc[i])[-2],list(data.iloc[i])[-1],self.a1,self.a2)
            optimizer.register(
                               params=params,
                               target=target,
                              )
        
        next_point_to_probe = optimizer.suggest(utility)
        points = list(next_point_to_probe.values())
        if num_variables == 1:
            points.append(0)
        points = [ round(elem,5) for elem in points]
        U = [str(x) for x in points]
        with open('input.json', 'r') as f:
            data = json.load(f)
            elements = list(data["general_flags"]["ldau_luj"].keys())
            for i in range(num_variables):
                data["general_flags"]["ldau_luj"][elements[i]]["U"] = float(U[i])
            f.close()
        
        with open('input.json', 'w') as f:
            json.dump(data,f,indent=4)
            f.close()

        return target

   
def calculate(command, outfilename, method):
    olddir = os.getcwd()
    calc = vasp_init(olddir+'/input.json')
    calc.init_atoms()

    calc.generate_input(olddir+'/%s/scf' %method,'scf','pbe')
    calc.generate_input(olddir+'/%s/band' %method,'band','pbe')
    try:
        os.chdir(olddir+'/%s/scf' %method)
        errorcode_scf = subprocess.call('%s > %s' %(command, outfilename), shell=True)
        os.system('cp CHG* WAVECAR %s/%s/band' %(olddir, method))
    finally:
        os.chdir(olddir+'/%s/band' %method)
        errorcode_band = subprocess.call('%s > %s' %(command, outfilename), shell=True)
        os.chdir(olddir)




