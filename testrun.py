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

warnings.simplefilter('ignore')

reppath = os.environ.get("SIMULATOR", '-1')
if reppath == '-1':
    reppath = os.getcwd()
    os.environ["SIMULATOR"] = reppath

n_objects = 2
spectrumgrid = [1, 1]
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
nproc = 2

# Setting up DataList for BALROG
for gp in simulation.grid:
    for k in selected_sig[gp.name]:
        for det in selected_sig[gp.name][k]:
            det_bl[gp.name][k][det] = drm.BALROGLike.from_spectrumlike(gp.response_generator[det][int(k[0]),int(k[1])]['generator'], rsp_time, simulation.det_rsp[det], free_position = True)
            datadict[gp.name][k] = DataList(*det_bl[gp.name][k].values())

print(len(datadict))


def flatten(d, parent_key=''):
    items = []
    for  k, v in d.items():
        new_key = (parent_key,  k) if parent_key else k
        try:
            items.extend(flatten(v, new_key).items())
        except:
            items.append((new_key,v))
    return dict(items)

fltdatadict=flatten(datadict)
enumdatanames=dict(enumerate(fltdatadict))
'''{0: ('gp1', ('0', '0')), 1: ('gp0', ('0', '0')), 2: ('gp1', ('1', '1')), 3: ('gp0', ('1', '1')), 4: ('gp1', ('1', '0')), 5: ('gp0', ('1', '0')), 6: ('gp0', ('0', '1')), 7: ('gp1', ('0', '1'))}'''
enumdatavalues=dict(enumerate(fltdatadict.values()))
'''{0: <threeML.data_list.DataList object at 0x7fcccf5aacd0>, 1: <threeML.data_list.DataList object at 0x7fccc2016d50>, 2: <threeML.data_list.DataList object at 0x7fcccf5aa090>, 3: <threeML.data_list.DataList object at 0x7fccbd48e4d0>, 4: <threeML.data_list.DataList object at 0x7fccc2da9dd0>, 5: <threeML.data_list.DataList object at 0x7fccbd4988d0>, 6: <threeML.data_list.DataList object at 0x7fccbd498750>, 7: <threeML.data_list.DataList object at 0x7fcccf5aadd0>}
'''
proc_advisor_dict={}
for i in enumdatanames.keys():
    proc_advisor_dict[i] = {'proc_list' : range(i*nproc, i*nproc+nproc), 'name' : enumdatanames[i]}
print(proc_advisor_dict)

for i in range(nproc*spectrumgrid[0]*spectrumgrid[1]*n_objects):
    grb_number = i//nproc

    if rank//nproc == grb_number:
        data = enumdatavalues[grb_number]

        ra = 10.
        dec = 10.
        name=enumdatanames[grb_number][0]+enumdatanames[grb_number][1][0]+enumdatanames[grb_number][1][1]
        print('Thread '+str(rank)+' working on GRB '+name)

        cpl = Cutoff_powerlaw(piv=100.)
        cpl.K.prior = Log_uniform_prior()
        cpl.K.prior.lower_bound = 1e-3
        cpl.K.prior.upper_bound = 1e3
        cpl.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
        cpl.index.set_uninformative_prior(Uniform_prior)
        model = Model(PointSource(name, ra, dec, spectral_shape=cpl))
        bayes = BayesianAnalysis(model, data)
        wrap = [0]*len(model.free_parameters)
        wrap[0] = 1

        _ = bayes.sample_multinest(600, chain_name='chains_'+name+'/',
                                   importance_nested_sampling=False,
                                   const_efficiency_mode=False,
                                   wrapped_params=wrap,
                                   verbose=True,
                                   resume=False)
        if rank%nproc == 0:
            bayes.results.write_to('location_results_'+name+'.fits', overwrite = True)
