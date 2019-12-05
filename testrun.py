import warnings
from collections import defaultdict, MutableMapping
import os
import operator
from Simulator import *
import numpy as np
import chainconsumer
from trigdat_reader import TrigReader
from mpi4py import MPI
from glob import glob
mpi = MPI.COMM_WORLD
rank = mpi.Get_rank()
size=mpi.Get_size()















warnings.simplefilter('ignore')

reppath = os.environ.get("SIMULATOR", '-1')
if reppath == '-1':
    reppath = os.getcwd()
    os.environ["SIMULATOR"] = reppath

n_objects = 2
spectrumgrid = [1, 1]
ngrbs=spectrumgrid[0]*spectrumgrid[1]*n_objects
trigdat = glob("rawdata/131229277/glg_trigdat_all_bn131229277_v0*.fit")[0]
simulation = Simulator(n_objects, spectrumgrid, trigdat)
det_list = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5','n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']
det_bl = defaultdict(lambda: defaultdict(dict))
rsp_time = 0.
n_detectors = 3

if rank == 0:
    simulation.setup(algorithm='Fibonacci', irange=[-1.6, -1], crange=[50, 150], K=50, background_function=Powerlaw(K=10., piv=100))
    # simulation.coulomb_refining(1000)
    simulation.generate_j2000()
    os.chdir(reppath)
    simulation.generate_DRM_spectrum(trigger="131229277", save=True)

    #for counter, gp in enumerate(simulaiton.grid):
        


    det_sig = dict()

    full_sig = {gp.name : {(str(i), str(j)): {det : gp.response_generator[det][i, j]["significance"] for det in det_list} for (i, j), value in np.ndenumerate(gp.value_matrix)} for gp in simulation.grid}
    selected_sig = {}
    min_sig = {}
    for (gp_key, gp_value) in full_sig.items():
        temp_selected_sig=dict()
        temp_min_sig=dict()
        for ij_key in gp_value:
            ls=[]
            lsval=[]
            for tuple in sorted(full_sig[gp_key][ij_key].items(), key=operator.itemgetter(1))[-n_detectors:]:
                ls.append(tuple[0])
                lsval.append(tuple[1])
            temp_selected_sig[ij_key]=ls
            temp_min_sig[ij_key]=min(lsval)
        selected_sig[gp_key]=temp_selected_sig
        min_sig[gp_key]=temp_min_sig

    det_list_new = []
    os.chdir(reppath)
    print(os.getcwd())
    f= open("radec.txt", 'w')
    with open('params.csv', 'w') as csvf:
        for x in simulation.grid[0].value_matrix:
            np.savetxt(csvf, x, header='K,xc,index', fmt='%f,%f,%f', delimiter=",", comments='')
    for gp in simulation.grid:
        f.write("=========\n")
        f.write(gp.name+":\n")
        f.write('RA: ' + str(gp.ra)+'\n')
        f.write('DEC: ' + str(gp.dec)+'\n')
        f.write("=========\n")


        for k in selected_sig[gp.name]:

            f.write('Entry ('+k[0]+","+k[1]+")\n")
            f.write('Selected Detectors: ')
            f.write('['+",".join(selected_sig[gp.name][k])+']\n')
            f.write('Min Significance: '+ str(min_sig[gp.name][k])+"\n\n")

            for det in selected_sig[gp.name][ij_key]:
                if det != 'b0' and det != 'b1':
                    gp.response_generator[det][int(k[0]),int(k[1])]['generator'].set_active_measurements('8.1-900')
                else:
                    gp.response_generator[det][int(k[0]),int(k[1])]['generator'].set_active_measurements('250-30000')

    coord_list = []
    for gp in simulation.grid:
        coord_list.append(gp.coord)

else:
    coord_list = []
    selected_sig = {}

# Setting up for every other core
coord_list = mpi.bcast(coord_list, root=0)
selected_sig = mpi.bcast(selected_sig, root=0)

if not mpi.rank == 0:
    simulation.setup(algorithm='Fibonacci', irange=[-1.6, -1], crange=[50, 150], K=50, background_function=Powerlaw(K=10, piv=100))
i = 0

for gp in simulation.grid:
    gp.update_coord(coord_list[i])
    i += 1

os.chdir(reppath)
simulation.generate_j2000()
simulation.load_DRM_spectrum()

point = simulation.grid[0]
det_sig = dict()
datadict = defaultdict(lambda: defaultdict(dict))
if size>ngrbs:
    nproc = size//ngrbs
else:
    print("Less processors than GRBs")

# Setting up DataList for BALROG

