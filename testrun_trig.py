import os
from Simulator import *
import numpy as np
import chainconsumer
from trigdat_reader import TrigReader
import pymultinest as pmn

import warnings
warnings.simplefilter('ignore')


n_objects=1
spectrumgrid=[1,1]
#trigfile="rawdata/191017391/glg_trigdat_all_bn191017391_v01.fit"
trigfile="/home/niklasvm/Envs/Simulator/rawdata/glg_trigdat_all_bn190504678_v01.fit"
simulation=Simulator(n_objects,spectrumgrid,trigfile)
det_list=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']

simulation.setup(algorithm='Fibonacci',irange=[-1.6,-1],crange=[15,20],K=20)
# simulation.coulomb_refining(1000)
simulation.generate_j2000()
simulation.generate_TRIG_spectrum()
with open("radec.txt",'wb') as f:
    f.write('RA: '+ str(simulation.grid[0].ra)+'\n')
    f.write('DEC: '+ str(simulation.grid[0].dec))
with open('params.csv','w') as outfile:
    for x in simulation.grid[0].value_matrix:
        np.savetxt(outfile,x,fmt='%.5f,%f,%f',header='K,xc,index',delimiter=",",comments='')


det_bl=dict()
rsp_time=0.

det_list_new=[]
for det in det_list:
    if det != 'b0' and det != 'b1':
        simulation.grid[0].photon_counts[det][0,0].set_active_measurements('8.1-900')
    else:
        simulation.grid[0].photon_counts[det][0,0].set_active_measurements('250-30000')

    if simulation.grid[0].photon_counts[det][0,0].significance>100:
        det_list_new.append(det)

point=simulation.grid[0]


for det in det_list_new:
    det_bl[det]=drm.BALROGLike.from_spectrumlike(point.photon_counts[det][0,0],rsp_time,simulation.det_rsp[det],free_position=True)

data = DataList(*det_bl.values())

ra = 10.
dec = 10.

cpl = Cutoff_powerlaw()
cpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e3)

cpl.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
cpl.index.set_uninformative_prior(Uniform_prior)
model = Model(PointSource('grb',ra,dec,spectral_shape=cpl))
bayes = BayesianAnalysis(model, data)
wrap = [0]*len(model.free_parameters)
wrap[0] = 1
_ =bayes.sample_multinest(400,chain_name='chains/',
                        importance_nested_sampling=False,
                        const_efficiency_mode=False,
                        wrapped_params=wrap,
                        verbose=True,
                        resume=False)
bayes.results.write_to('location_results2.fits', overwrite=True) 

res=bayes.results
str_path=os.getcwd()

cc_plot=res.corner_plot_cc()
cc_plot.savefig(str_path+'/cc_plot_test.pdf')
spectrum_plot=display_spectrum_model_counts(bayes,step=False);
spectrum_plot.savefig(str_path+'/spectrum_plot_test.pdf')

