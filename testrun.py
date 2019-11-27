import os
import operator
from Simulator import *
import numpy as np
import chainconsumer
from trigdat_reader import TrigReader
from mpi4py import MPI
from glob import glob
mpi=MPI.COMM_WORLD
rank=mpi.Get_rank()

import warnings
warnings.simplefilter('ignore')

n_objects=1
spectrumgrid=[1,1]
trigdat=glob("rawdata/131229277/glg_trigdat_all_bn131229277_v0*.fit")[0]
simulation=Simulator(n_objects,spectrumgrid,trigdat)
det_list=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
det_bl=dict()
rsp_time=0.

if rank==0:
    simulation.setup(algorithm='Fibonacci',irange=[-1.6,-1],crange=[50,150],K=50,background_function=Powerlaw(K=20.))
    # simulation.coulomb_refining(1000)
    simulation.generate_j2000()
    simulation.generate_DRM_spectrum(trigger="131229277",save=True)
    with open('params.csv','w') as outfile:
        for x in simulation.grid[0].value_matrix:
            np.savetxt(outfile,x,fmt='%.5f,%f,%f',header='K,xc,index',delimiter=",",comments='')



    det_sig=dict()
    for det in det_list:
        det_sig[det]=simulation.grid[0].response_generator[det][0,0].significance
    det_list3=[]
    for tuple in sorted(det_sig.items(),key=operator.itemgetter(1))[-4:]: det_list3.append(tuple[0])

    det_list_new=[]
    for det in det_list3:
        if det != 'b0' and det != 'b1':
            simulation.grid[0].response_generator[det][0,0].set_active_measurements('8.1-900')
        else:
            simulation.grid[0].response_generator[det][0,0].set_active_measurements('250-30000')

    with open("radec.txt",'wb') as f:
        f.write('RA: '+ str(simulation.grid[0].ra)+'\n')
        f.write('DEC: '+ str(simulation.grid[0].dec)+'\n')
        f.write('Detectors: ')  
        f.write('['+",".join(det_list3)+']')

    coord_list=[]
    for gp in simulation.grid:
        coord_list.append(gp.coord)

else:
    coord_list=[]
    det_list3=[]

coord_list=mpi.bcast(coord_list,root=0)
det_list3=mpi.bcast(det_list3,root=0)
if not mpi.rank==0:
    simulation.setup(algorithm='Fibonacci',irange=[-1.6,-1],crange=[50,150],K=50,background_function=Powerlaw(K=20))
i=0
for gp in simulation.grid:
    gp.update_coord(coord_list[i])
    i+=1
os.chdir("/home/niklasvm/Envs/Simulator/GBMGridSimulator")
simulation.generate_j2000()

simulation.load_DRM_spectrum()
point=simulation.grid[0]
det_sig=dict()
for det in det_list:
    det_sig[det]=simulation.grid[0].response_generator[det][0,0].significance
print(det_sig)
for det in det_list3:
    det_bl[det]=drm.BALROGLike.from_spectrumlike(point.response_generator[det][0,0],rsp_time,simulation.det_rsp[det],free_position=True)

data = DataList(*det_bl.values())

ra = 10.
dec = 10.

cpl = Cutoff_powerlaw(piv=100.)
cpl.K.prior = Log_uniform_prior()
cpl.K.prior.lower_bound=1e-3
cpl.K.prior.upper_bound=1e3
cpl.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
cpl.index.set_uninformative_prior(Uniform_prior)
model = Model(PointSource('grb',ra,dec,spectral_shape=cpl))
bayes = BayesianAnalysis(model, data)
wrap = [0]*len(model.free_parameters)
wrap[0] = 1

_ =bayes.sample_multinest(600,chain_name='chains/',
                        importance_nested_sampling=False,
                        const_efficiency_mode=False,
                        wrapped_params=wrap,
                        verbose=True,
                        resume=False)
if rank==0:
    bayes.results.write_to('location_results2.fits', overwrite=True) 

    res=bayes.results
    str_path=os.getcwd()

    cc_plot=res.corner_plot_cc();
    cc_plot.savefig(str_path+'/cc_plot_test.pdf')
    spectrum_plot=display_spectrum_model_counts(bayes,step=False);
    spectrum_plot.savefig(str_path+'/spectrum_plot_test.pdf')

