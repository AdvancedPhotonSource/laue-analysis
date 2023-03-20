#!/usr/bin/env python3

from re import M
from tkinter import N
import cold
import fire
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage, stats, optimize, interpolate
from matplotlib.gridspec import GridSpec
import os
import matplotlib
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 9, 'font.family' : 'times'})
from matplotlib.pyplot import cm
import warnings
warnings.filterwarnings('ignore')


def main(path, debug=False):
    """Runs the reconstruction workflow given parameters 
    in a configuration file.

    Parameters
    ----------
    path: string
        Path of the YAML file with configuration parameters.

    debug: bool
        If True, plots the fitted signals. 

    Returns
    -------
        None
    """

    def costfunc(vals, geo, dat, ind, algo, a):

        # Update optimization parameters
        geo['mask']['focus']['cenx'] = vals[0]
        geo['mask']['focus']['dist'] = vals[1]
        geo['mask']['focus']['anglez'] = vals[2]
        geo['mask']['focus']['angley'] = vals[3]
        geo['mask']['focus']['anglex'] = vals[4]
        geo['mask']['focus']['cenz'] = vals[5]
        geo['mask']['widening']  = vals[6]
        geo['mask']['thickness']  = vals[7]

        pos, sig, scl = cold.decode(dat, ind, comp, geo, algo, debug=debug)
        dap, lau = cold.resolve(dat, ind, pos, sig, geo, comp)

        # cost function evaluations for lau
        grid = geo['source']['grid']
        x = np.arange(*grid)

        indp0 = np.zeros((lau.shape[0]), dtype='int32')
        for m in range(lau.shape[0]):
            ha0 = np.cumsum(lau[m])
            if np.max(ha0) > 0:
                ha0 = ha0 / np.max(ha0)
                indp0[m] = np.min(np.where(ha0 >= 0.5)) 
                
        # cost function evaluations for lau
        cost = np.sum(np.power(x[indp0.flatten()], 2)) / lau.shape[0]
        cold.saveplt('tmp/autofocus-depth/dep/dep-' + str(a), dap, geo['source']['grid'])
        return cost

    indices = np.array(
            [
            [395, 11],
            [1766, 33],
            [1236, 164],
            [807, 198],
            [1232, 248],
            [1666, 256],
            [214, 337],
            [590, 342],
            [1878, 434],
            [1223, 450],
            [1216, 603],
            [487, 640],
            [784, 722],
            [1211, 723],
            [573, 747],
            [1959, 756],
            [1642, 791],
            [840, 844],
            [1204, 900],
            [1575, 905],
            [882, 939],
            [119, 985],
            [1525, 995],
            [189, 1017],
            [1200, 1025],
            [352, 1099],
            [1197, 1119],
            [469, 1166],
            [557, 1221],
            [1939, 1305],
            [681, 1307],
            [1843, 1346],
            [763, 1371],
            [1709, 1410],
            [1620, 1458],
            [1509, 1525],
            [1179, 1782],
            [542, 1817],
            [453, 1840],
            [167, 1943],
            ])

    # Optimization
    file, comp, geo, algo = cold.config(path)
    dat, ind = cold.load(file, index=indices) 

    # Sequence of coordinates
    seq = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype='int')

    # geo['mask']['focus']['cenx']
    # geo['mask']['focus']['dist']
    # geo['mask']['focus']['anglez']
    # geo['mask']['focus']['angley']
    # geo['mask']['focus']['anglex']
    # geo['mask']['focus']['cenz']
    # geo['mask']['widening'] 
    # geo['mask']['thickness'] 

    # Search regions
    lbound = np.array([0.5, 1.5, 44.5, -2.5, 2.0, 1.7, 4.0, 0.0]) 
    ubound = np.array([0.6, 1.9, 45.5, -1.5, 3.0, 2.7, 5.0, 0.0])
    mpoint = 0.5 * (lbound + ubound)

    # Parameter init
    vl = mpoint.copy()
    vu = mpoint.copy()
    vm = mpoint.copy()

    a = 0
    b = 0
    for nn in range(10): # For each CG iteration
        for qq in range(7): # For each coordinate

            # Lower bound cost
            vl[seq[qq]] = lbound[seq[qq]]
            costl = costfunc(vl, geo, dat, ind, algo, a)
            a += 1

            # Upper bound cost
            vu[seq[qq]] = ubound[seq[qq]]
            costu = costfunc(vu, geo, dat, ind, algo, a)
            a += 1

            for it in range(10): # For each binary search iteration
                
                # Middle point cost
                vm[seq[qq]] = (vu[seq[qq]] + vl[seq[qq]]) * 0.5
                costm = costfunc(vm, geo, dat, ind, algo, a)
                a += 1

                # Update points and bounds
                costlm = costl - costm
                costum = costu - costm
                if costlm > costum:
                    vl[seq[qq]] = vm[seq[qq]]
                    costl = costm
                else:
                    vu[seq[qq]] = vm[seq[qq]]
                    costu = costm
                vm[seq[qq]] = (vu[seq[qq]] + vl[seq[qq]]) * 0.5

                # Save/print results
                print (str(nn) + '-' + str(qq) + '-' + str(it) + '-' + str(a) + ': ' + str(vm[0:8]))

            # Save/print results
            if not os.path.exists('tmp/autofocus-convergence/conv'):
                os.makedirs('tmp/autofocus-convergence/conv')
            np.save('tmp/autofocus-convergence/conv' + '/conv-' + str(b) + '.npy', [vl, vm, vu, costl, costm, costu])
            b += 1
        
    if not os.path.exists('tmp/autofocus-results'):
        os.makedirs('tmp/autofocus-results')
    np.save('tmp/autofocus-results/results' + '.npy', vm)

if __name__ == '__main__':
    fire.Fire(main)

