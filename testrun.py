import warnings
warnings.simplefilter('ignore')
import os
os.chdir('/home/niklas/Dokumente/Bachelor/Python/PythonScripts/SimulatorSetup/')
from Simulator import *


n_objects=10
spectrumgrid=[4,4]
simulation=Simulator(n_objects,spectrumgrid)

simulation.setup(algorithm='Fibonacci',irange=[-1.6,-1],crange=[50,150],K=100)
trigdat="/home/niklas/Dokumente/Bachelor/rawdata/191017391/glg_trigdat_all_bn191017391_v01.fit"
simulation.generate_j2000(trigdat)
simulation.generate_DRM_spectrum()
