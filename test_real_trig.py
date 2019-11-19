from trigdat_reader import *
from threeML import *

import warnings                                                                                                                                
warnings.simplefilter('ignore')   # we load the trigdat file
trig1='/home/niklas/Dokumente/Bachelor/rawdata/190504678/glg_trigdat_all_bn190504678_v01.fit'
trig2='/home/niklasvm/Envs/Simulator/rawdata/131229227/glg_trigdat_all_bn131229277_v02.fit'
trig_reader = TrigReader('rawdata/131229227/glg_trigdat_all_bn131229277_v02.fit',fine=True,verbose=False)
trig_reader.set_background_selections('3-30','-10--5')
trig_reader.set_active_time_interval('-0.5-0.5')
# we choose which detectors to use and create the data plugin
det_list=['n6','n7','n8','n9','na','nb','b1']
trigdata  = trig_reader.to_plugin(*det_list)
#display count spectrum for the detectors
# create data list for our detector data
data_list = DataList(*trigdata)
# we define our model for the source. in this case a cur-off power-law
cpl = Cutoff_powerlaw()                                                   
                                                                                
cpl.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=100)      
cpl.xc.prior = Log_uniform_prior(lower_bound=1e-2, upper_bound=1e4)                                                                                       
cpl.index.set_uninformative_prior(Uniform_prior)

# we set up an initial value for the ra, dec (which one is not important, does not change the results of the fit)
ra, dec = 0,0
model = Model(PointSource('test',ra,dec,spectral_shape=cpl))
# wrap for ra angle
wrap = [0]*len(model.free_parameters)                                                                              
wrap[0] = 1    
bayes=BayesianAnalysis(model,data_list)
                           
# we use multinest to sample the posterior dsistribution(best to run this in parallel and not in a notebook)
_ =bayes.sample_multinest(500,
                            chain_name='/home/niklasvm/Envs/Simulator/chains/test', 
                            importance_nested_sampling=False,                                                                                  
                            const_efficiency_mode=False,                                                                                       
                            wrapped_params=wrap,                                                                                               
                            verbose=True,                                                                                                      
                            resume=False) 
                                                                                
res=bayes.results.write_to('location_real_trig.fits',overwrite=True)
