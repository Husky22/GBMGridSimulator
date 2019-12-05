from Simulator import *
import matplotlib
from mpi4py import MPI
import os
from warnings import simplefilter
simplefilter("ignore")
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
envpath=os.environ.get('SIMULATOR')
simulation=Simulator(2,[1,1],envpath)
simulation.setup(K=1E-6)
simulation.generate_j2000()
if rank==0:
    simulation.grid_generate_DRM_spectrum(snr=20)
    simulation.save_DRM_spectra()

comm.Barrier()
if rank!=0:
    simulation.load_DRM_spectra()

simulation.run()