import os
import json
import bayes_opt
import subprocess
import numpy as np
import pandas as pd
import pymatgen as mg
import xml.etree.ElementTree as ET

from ase import Atoms, Atom
from ase.calculators.vasp.vasp import Vasp
from ase.dft.kpoints import *

from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from string import ascii_lowercase
from BayesOpt4dftu.special_kpath import kpath_dict

from vaspvis import Band
from vaspvis.utils import get_bandgap

from matplotlib import pyplot as plt
from matplotlib import cm, gridspec

# TODO: 1. SCF calculation in DFT+U missing U tags in INCAR.
#       2. Check whether the U value has an duplicate in u.txt.
#       3. Modify the BO for multi-U condition (More than 2 U values need to be optimized).
#       4. Fix the bug that code output incorrect U in when U values are optimized for elements without the first one.


def readgap(vasprun, kpoints):
    run = BSVasprun(vasprun)
    bs = run.get_band_structure(kpoints)
    if (bs.is_metal() == False):
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
            poscar[7] = 'Direct\n'
            f.close()

        with open(path + '/POSCAR', 'w') as d:
            d.writelines(poscar)
            d.close()

    def kpt4pbeband(self, path, import_kpath):
        if import_kpath:
            special_kpoints = kpath_dict
        else:
            special_kpoints = get_special_points(self.atoms.cell)

        num_kpts = self.struct_info['num_kpts']
        labels = self.struct_info['kpath']
        kptset = list()
        lbs = list()
        if labels[0] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[0]])
            lbs.append(labels[0])

        for i in range(1, len(labels)-1):
            if labels[i] in special_kpoints.keys():
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
        if labels[-1] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[-1]])
            lbs.append(labels[-1])

        # Hardcoded for EuS and EuTe since one of the k-point is not in the special kpoints list.
        if 'EuS' in self.atoms.symbols or 'EuTe' in self.atoms.symbols:
            kptset[0] = np.array([0.5, 0.5, 1])

        kpt = Kpoints(comment='band', kpts=kptset, num_kpts=num_kpts,
                      style='Line_mode', coord_type="Reciprocal", labels=lbs)
        kpt.write_file(path+'/KPOINTS')

    def kpt4hseband(self, path, import_kpath):
        ibz = open(path+'/IBZKPT', 'r')
        num_kpts = self.struct_info['num_kpts']
        labels = self.struct_info['kpath']
        ibzlist = ibz.readlines()
        ibzlist[1] = str(num_kpts*(len(labels)-1) +
                         int(ibzlist[1].split('\n')[0])) + '\n'
        if import_kpath:
            special_kpoints = kpath_dict
        else:
            special_kpoints = get_special_points(self.atoms.cell)
        for i in range(len(labels)-1):
            k_head = special_kpoints[labels[i]]
            k_tail = special_kpoints[labels[i+1]]
            increment = (k_tail-k_head)/(num_kpts-1)
            ibzlist.append(' '.join(map(str, k_head)) +
                           ' 0 ' + labels[i] + '\n')
            for j in range(1, num_kpts-1):
                k_next = k_head + increment*j
                ibzlist.append(' '.join(map(str, k_next)) + ' 0\n')
            ibzlist.append(' '.join(map(str, k_tail)) +
                           ' 0 ' + labels[i+1] + '\n')
        with open(path+'/KPOINTS', 'w') as f:
            f.writelines(ibzlist)

    def generate_input(self, directory, step, xc, import_kpath):
        flags = {}
        flags.update(self.general_flags)
        flags.update(self.input_dict[step])
        if step == 'scf':
            if xc == 'pbe':
                flags.update(self.input_dict[xc])
            calc = Vasp(self.atoms, directory=directory,
                            kpts=self.struct_info['kgrid_'+xc], gamma=True, **flags)
            calc.write_input(self.atoms)
            if str(self.atoms.symbols) in ['Ni2O2']:
                mom_list = {'Ni': 2, 'Mn': 5, 'Co': 3, 'Fe': 4}
                s = str(self.atoms.symbols[0])
                incar_scf = Incar.from_file(directory+'/INCAR')
                incar_scf['MAGMOM'] = '%s -%s 0 0' % (mom_list[s], mom_list[s])
                incar_scf.write_file(directory+'/INCAR')

            self.modify_poscar(path=directory)
        elif step == 'band':
            flags.update(self.input_dict[xc])
            calc = Vasp(self.atoms, directory=directory, gamma=True, **flags)
            calc.write_input(self.atoms)
            self.modify_poscar(path=directory)
            if xc == 'pbe':
                self.kpt4pbeband(directory, import_kpath)
            elif xc == 'hse':
                print(directory)
                self.kpt4hseband(directory, import_kpath)


class delta_band(object):
    def __init__(self, bandrange=10, path='./', iteration=1, interpolate=False):
        self.path = path
        self.br = bandrange
        self.interpolate = interpolate
        self.vasprun_hse = os.path.join(path, 'hse/band/vasprun.xml')
        self.kpoints_hse = os.path.join(path, 'hse/band/KPOINTS')
        self.vasprun_dftu = os.path.join(path, 'dftu/band/vasprun.xml')
        self.kpoints_dftu = os.path.join(path, 'dftu/band/KPOINTS')
        self.iteration = iteration

    def readInfo(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        ispin = int(root.findall(
            './parameters/separator/.[@name="electronic"]/separator/.[@name="electronic spin"]/i/.[@name="ISPIN"]')[0].text)
        nbands = int(root.findall(
            './parameters/separator/.[@name="electronic"]/i/.[@name="NBANDS"]')[0].text)
        nkpts = len(root.findall('./kpoints/varray/.[@name="kpointlist"]/v'))

        return ispin, nbands, nkpts

    def access_eigen(self, b, interpolate=False):
        wave_vectors = b._get_k_distance()
        eigenvalues = b.eigenvalues

        if interpolate:
            _, eigenvalues_interp = b._get_interpolated_data(
                wave_vectors=wave_vectors,
                data=eigenvalues
            )

        if interpolate:
            return eigenvalues_interp
        else:
            return eigenvalues

    def locate_and_shift_bands(self, eigenvalues):
        band_mean = eigenvalues.mean(axis=1)

        below_index = np.where(band_mean < 0)[0]
        above_index = np.where(band_mean >= 0)[0]

        vbm = np.max(eigenvalues[below_index])
        cbm = np.min(eigenvalues[above_index])

        if cbm < vbm:
            vbm = 0.0
            cbm = 0.0

        valence_bands = eigenvalues[below_index[-self.br:]]
        conduction_bands = eigenvalues[above_index[:self.br]]

        valence_bands -= vbm
        conduction_bands -= cbm

        shifted_bands = np.r_[conduction_bands, valence_bands]

        return shifted_bands

    def deltaBand(self):
        ispin_hse, nbands_hse, nkpts_hse = self.readInfo(self.vasprun_hse)
        ispin_dftu, nbands_dftu, nkpts_dftu = self.readInfo(self.vasprun_dftu)

        
        if nbands_hse != nbands_dftu:
            raise Exception('The band number of HSE and GGA+U are not match!')

        kpoints = [line for line in open(self.kpoints_hse) if line.strip()]
        kpts_diff = 0
        for ii, line in enumerate(kpoints[3:]):
            if line.split()[3] != '0':
                kpts_diff += 1

        if nkpts_hse - kpts_diff != nkpts_dftu:
            raise Exception(
                'The kpoints number of HSE and GGA+U are not match!')

        new_n = 500

        if ispin_hse == 1 and ispin_dftu == 1:
            band_hse = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )
            band_dftu = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
                bandgap=True,
                printbg=False,
            )

            eigenvalues_hse = self.access_eigen(band_hse, interpolate=self.interpolate)
            eigenvalues_dftu = self.access_eigen(band_dftu, interpolate=self.interpolate)

            shifted_hse = self.locate_and_shift_bands(eigenvalues_hse)
            shifted_dftu = self.locate_and_shift_bands(eigenvalues_dftu)

            n = shifted_hse.shape[0] * shifted_hse.shape[1]
            delta_band = sum((1/n)*sum((shifted_hse - shifted_dftu)**2))**(1/2)

            bg = get_bandgap(
                folder=os.path.join(self.path, 'dftu/band'),
                printbg=False,
                method=1,
                spin='both',
            )

            incar = Incar.from_file('./dftu/band/INCAR')
            u = incar['LDAUU']
            u.append(bg)
            u.append(delta_band)
            output = ' '.join(str(x) for x in u)

            with open('u_tmp.txt', 'a') as f:
                f.write(output + '\n')
                f.close

            return delta_band

        elif ispin_hse == 2 and ispin_dftu == 2:
            band_hse_up = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            band_dftu_up = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
                bandgap=True,
                printbg=False,
            )

            band_hse_down = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            band_dftu_down = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            eigenvalues_hse_up = self.access_eigen(band_hse_up, interpolate=self.interpolate)
            eigenvalues_dftu_up = self.access_eigen(band_dftu_up, interpolate=self.interpolate)

            shifted_hse_up = self.locate_and_shift_bands(eigenvalues_hse_up)
            shifted_dftu_up = self.locate_and_shift_bands(eigenvalues_dftu_up)

            n_up = shifted_hse_up.shape[0] * shifted_hse_up.shape[1]
            delta_band_up = sum((1/n_up)*sum((shifted_hse_up - shifted_dftu_up)**2))**(1/2)

            eigenvalues_hse_down = self.access_eigen(band_hse_down, interpolate=self.interpolate)
            eigenvalues_dftu_down = self.access_eigen(band_dftu_down, interpolate=self.interpolate)

            shifted_hse_down = self.locate_and_shift_bands(eigenvalues_hse_down)
            shifted_dftu_down = self.locate_and_shift_bands(eigenvalues_dftu_down)

            n_down = shifted_hse_down.shape[0] * shifted_hse_down.shape[1]
            delta_band_down = sum((1/n_down)*sum((shifted_hse_down - shifted_dftu_down)**2))**(1/2)

            delta_band = np.mean([delta_band_up, delta_band_down])

            bg = get_bandgap(
                folder=os.path.join(self.path, 'dftu/band'),
                printbg=False,
                method=1,
                spin='both',
            )

            incar = Incar.from_file('./dftu/band/INCAR')
            u = incar['LDAUU']

            u.append(bg)
            u.append(delta_band)
            output = ' '.join(str(x) for x in u)

            with open('u_tmp.txt', 'a') as f:
                f.write(output + '\n')
                f.close

            return delta_band
        else:
            raise Exception('The spin number of HSE and GGA+U are not match!')


class bayesOpt_DFTU(object):
    def __init__(self, path, opt_u_index=(1, 1), u_range=(0, 10), kappa=2.5, alpha_1=1, alpha_2=1):
        self.input = path + 'u_tmp.txt'
        self.gap = readgap(path + '/hse/band/vasprun.xml',
                           path + '/hse/band/KPOINTS')
        self.kappa = kappa
        self.a1 = alpha_1
        self.a2 = alpha_2
        self.opt_u_index = np.array(opt_u_index) > 0
        self.u_range = u_range
        self.elements = {}

    def loss(self, y, y_hat, delta_band, alpha_1, alpha_2):
        return -alpha_1 * (y - y_hat) ** 2 - alpha_2 * delta_band ** 2
    
    def get_optimizer(self):
        data = pd.read_csv(self.input, header=0,
                           delimiter="\s", engine='python')
        num_rows, d = data.shape
        num_variables = sum(self.opt_u_index)
        if num_variables > 2:
            raise ValueError("BO larger than 2D are not supported yet!")
        variables_string = ascii_lowercase[:num_variables]
        pbounds = {}
        if num_variables == 1:
            pbounds[variables_string[0]] = self.u_range
        elif num_variables == 2:
            for variable in variables_string:
                pbounds[variable] = self.u_range
        utility_function = UtilityFunction(kind="ucb", kappa=self.kappa, xi=0)
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        
        for i in range(num_rows):
            values = list()
            for j in range(len(self.opt_u_index)):
                if self.opt_u_index[j]:
                    values.append(data.iloc[i][j])
            params = {}
            for (value, variable) in zip(values, variables_string):
                params[variable] = value
            target = self.loss(self.gap, list(
                data.iloc[i])[-2], list(data.iloc[i])[-1], self.a1, self.a2)

            optimizer.register(
                params=params,
                target=target,
            )
        
        return utility_function, optimizer, target

    def plot_bo(self, ratio=1):
        utility_function, optimizer, target = self.get_optimizer()
        plot_size = len(optimizer.res)*ratio
        opt_eles = [ele for i, ele in enumerate(self.elements) if self.opt_u_index[i]]

        if sum(self.opt_u_index) == 1:
            x = np.linspace(self.u_range[0], self.u_range[1], 10000).reshape(-1, 1)
            x_obs = np.array([res["params"]['a'] for res in optimizer.res]).reshape(-1,1)[:plot_size]
            y_obs = np.array([res["target"] for res in optimizer.res])[:plot_size]

            mu, sigma = posterior(optimizer, x_obs, y_obs, x)

            fig = plt.figure()
            gs = gridspec.GridSpec(2, 1) 
            axis = plt.subplot(gs[0])
            acq = plt.subplot(gs[1])
            axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
            axis.plot(x, mu, '--', color='k', label='Prediction')
            axis.fill(np.concatenate([x, x[::-1]]), 
                    np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
                alpha=.6, fc='c', ec='None', label='95% confidence interval')
            
            axis.set_xlim(self.u_range)
            axis.set_ylim((None, None))
            axis.set_ylabel('f(x)')

            acq.plot(x, utility_function, label='Acquisition Function', color='purple')
            acq.plot(x[np.argmax(utility_function)], np.max(utility_function), '*', markersize=15, 
                    label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
            acq.set_xlim(self.u_range)
            acq.set_ylim((np.min(utility_function)-0.5,np.max(utility_function)+0.5))
            acq.set_ylabel('Acquisition')
            acq.set_xlabel('U (eV)')
            axis.legend(loc=4, borderaxespad=0.)
            acq.legend(loc=4, borderaxespad=0.)

            plt.savefig('1D_kappa_%s_a1_%s_a2_%s.png' %(self.kappa, self.a1, self.a2), dpi = 400)

        if sum(self.opt_u_index) == 2:
            x = y = np.linspace(self.u_range[0], self.u_range[1], 300)
            X, Y = np.meshgrid(x, y)
            x = X.ravel()
            y = Y.ravel()
            X = np.vstack([x, y]).T[:, [1, 0]]

            x1_obs = np.array([[res["params"]["a"]] for res in optimizer.res])[:plot_size]
            x2_obs = np.array([[res["params"]["b"]] for res in optimizer.res])[:plot_size]
            y_obs = np.array([res["target"] for res in optimizer.res])[:plot_size]
            obs = np.column_stack((x1_obs, x2_obs))

            optimizer._gp.fit(obs, y_obs)
            mu, sigma = optimizer._gp.predict(X, eval)

            fig, axis = plt.subplots(1, 2, figsize=(15,5))
            plt.subplots_adjust(wspace = 0.2)
            
            axis[0].plot(x1_obs, x2_obs, 'D', markersize=4, color='k', label='Observations')
            axis[0].set_title('Gaussian Process Predicted Mean',pad=10)
            im1 = axis[0].hexbin(y, x, C=mu, cmap=cm.jet, bins=None)
            axis[0].axis([x.min(), x.max(), y.min(), y.max()])
            axis[0].set_xlabel('U_%s (eV)' %opt_eles[0],labelpad=5)
            axis[0].set_ylabel('U_%s (eV)' %opt_eles[1],labelpad=10,va='center')
            cbar1 = plt.colorbar(im1, ax = axis[0])

            utility = utility_function.utility(X, optimizer._gp, optimizer.max)
            axis[1].plot(x1_obs, x2_obs, 'D', markersize=4, color='k', label='Observations')
            axis[1].set_title('Acquisition Function',pad=10)
            axis[1].set_xlabel('U_%s (eV)' %opt_eles[0],labelpad=5)
            axis[1].set_ylabel('U_%s (eV)' %opt_eles[1],labelpad=10,va='center')
            im3 = axis[1].hexbin(y, x, C=utility, cmap=cm.jet, bins=None)
            axis[1].axis([x.min(), x.max(), y.min(), y.max()])
            cbar3 = plt.colorbar(im3, ax = axis[1])

            plt.savefig('2D_kappa_%s_a1_%s_a2_%s.png' %(self.kappa, self.a1, self.a2), dpi = 400)


    def bo(self):
        utility_function, optimizer, target = self.get_optimizer()
        next_point_to_probe = optimizer.suggest(utility_function)

        points = list(next_point_to_probe.values())
        points = [round(elem, 6) for elem in points]

        U = [str(x) for x in points]
        with open('input.json', 'r') as f:
            data = json.load(f)
            elements = list(data["pbe"]["ldau_luj"].keys())
            self.elements = elements
            for i in range(len(self.opt_u_index)):
                if self.opt_u_index[i]:
                    try:
                        data["pbe"]["ldau_luj"][elements[i]
                                                ]["U"] = round(float(U[i]), 6)
                    except:
                        data["pbe"]["ldau_luj"][elements[i]
                                                ]["U"] = round(float(U[i-1]), 6)
            f.close()

        with open('input.json', 'w') as f:
            json.dump(data, f, indent=4)
            f.close()

        return target


def calculate(command, outfilename, method, import_kpath):
    olddir = os.getcwd()
    calc = vasp_init(olddir+'/input.json')
    calc.init_atoms()

    if method == 'dftu':
        calc.generate_input(olddir+'/%s/scf' %
                            method, 'scf', 'pbe', import_kpath)
        calc.generate_input(olddir+'/%s/band' %
                            method, 'band', 'pbe', import_kpath)

    if os.path.isfile(f'{olddir}/{method}/band/eigenvalues.npy'):
        os.remove(f'{olddir}/{method}/band/eigenvalues.npy')

    elif method == 'hse':
        calc.generate_input(olddir+'/%s/scf' %
                            method, 'scf', 'hse', import_kpath)
        if not os.path.exists(olddir+'/%s/band' % method):
            os.mkdir(olddir+'/%s/band' % method)

    try:
        os.chdir(olddir+'/%s/scf' % method)
        errorcode_scf = subprocess.call(
            '%s > %s' % (command, outfilename), shell=True)
        os.system('cp CHG* WAVECAR IBZKPT %s/%s/band' % (olddir, method))
        if method == 'hse':
            calc.generate_input(olddir+'/%s/band' %
                                method, 'band', 'hse', import_kpath)
    finally:
        os.chdir(olddir+'/%s/band' % method)
        errorcode_band = subprocess.call(
            '%s > %s' % (command, outfilename), shell=True)
        os.chdir(olddir)
