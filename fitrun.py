from Simulator import *
import matplotlib
from mpi4py import MPI
import os
from warnings import simplefilter
simplefilter("ignore")
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
envpath=os.environ.get('SIMULATOR')
simulation=Simulator(100,[1,1],envpath)
simulation.setup(K=1E-6)
simulation.generate_j2000()
simulation.grid_generate_DRM_spectrum(snr=20)

comm.Barrier()
simulation.load_DRM_spectra()

simulation.run()
