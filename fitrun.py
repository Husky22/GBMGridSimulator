from Simulator import *
import matplotlib
from warnings import simplefilter
simplefilter("ignore")
simulation=Simulator(2,[1,1],"/home/niklas/venv/GBMGridSimulator/rawdata/191017391/glg_trigdat_all_bn191017391_v01.fit")
simulation.setup(K=1E-6)
simulation.generate_j2000()
simulation.grid_generate_DRM_spectrum(snr=20)
simulation.run()
