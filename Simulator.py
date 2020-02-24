import csv
import gc
import json
import math
import operator
import os
import random
import copy

import astropy.coordinates as coord
import astropy.units as u
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from mpi4py import MPI
from sphere.distribution import fb83
from tabulate import tabulate
import threeML as tml
import spherical_geometry as spg
from sympy.utilities.iterables import multiset_permutations

import gbm_drm_gen as drm
from gbmgeometry import GBM, GBMFrame, PositionInterpolator, gbm_frame
from glob import glob

mpl.use('Agg')
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

class SimulationObj(object):
    pass



class Simulator(SimulationObj):

    """
    Fermi GBM Simulator

    """
    global det_list
    det_list = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5',
                'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

    def __init__(self, name, source_number, spectrum_matrix_dimensions, directory, trigger = "131229277"):
        """
        A Simulation instance containing the grid.

        Instance serves for:
         - Grid creation 
        >>> Simulator.setup(...)
         - Coordinate transformation
        >>> Simulator.add_j2000(...)
         - Spectra generation
        >>> Simulator.grid_generate_DRM_spectrum(...)
         - Fisher sample creation
        >>> Simulator.create_fisher_samples(...)
         - Bulk fitting of true and fisher points
        >>> Simulator.refit_spectra(...)

        Parameters
        ----------
        name : string
            Name of the simulation to specify folder name to save the generated data
            "FitSimulation1" results in the folder "FitSimulation1_Files"

        source_number : int
            Specifies the number of GridPoints (GRBs) to create

        spectrum_matrix_dimensions : tuple
            Specify the number of different spectral parameters to simulate 
            (3,4) corresponds to 3 different index values and 4 different E_peak values

        directory : string
            Should contain the GBMGridSimulator path

        trigger : string
            Define the trigger to use for GBM position interpolation
            looks for trigdat file in "directory/rawdata/trigger"
            131229277 comes with GitHub Repository, others have to be downloaded

        """
        self.simulation_name = name
        self.N = source_number
        self.spectrum_dimension = spectrum_matrix_dimensions # list
        self.grid = None
        self.j2000_generate = False
        self.indexrange = None
        self.cutoffrange = None
        SimulationObj.directory = directory
        self.trigger = trigger
        self.trigger_folder = self.directory+"/rawdata/"+self.trigger
        self.trigfile = glob(self.trigger_folder + '/glg_trigdat_all_bn'+self.trigger+'_v0*.fit')[0]

    def setup(self, K, irange=[-2, -1], crange=[100, 400], algorithm='Fibonacci', skeleton=False ):
        '''
        Setup the GRB Grid

        Setup the GridPoints with its spectrum matrices and the background function for your Simulation.
        Create a DRMGen Response Generator and calculate the RA and DEC for your GridPoints

        Parameters
        ----------
        K : float
            initial Amplitude of the Spectrum
        irange : float list
            index range
        crange : float list
            cutoff range
        algorithm : string
            Specify used algorithm for GridPoint distribution
            'Fibonacci' is the only one available 
        skeleton : boolean
            Skeleton deactivates HDF5 saving

        '''
        SimulationObj.skeleton = skeleton
        SimulationObj.sim_path=self.directory+"/"+self.simulation_name+"_Files/"
        if rank == 0:
            
            print("Garbage Collection enabled: "+str(gc.isenabled())+"\n")
            if skeleton is False:
                if not os.path.exists(self.sim_path):
                    os.makedirs(self.sim_path)
                SimulationObj.simulation_file_path=self.sim_path+self.simulation_name+".hdf5"
                #  for obj in gc.get_objects():
                #      if isinstance(obj,h5py.File):
                #          try:
                #              obj.close()
                #              simulation_file = h5py.File(self.simulation_file_path,"r+")
                #          except:
                #              pass
                #          h5file_exists=True
                #      else: h5file_exists=False

                simulation_file = h5py.File(self.simulation_file_path,"a")
                simulation_file.create_group("grid")
                simulation_file.close()

            else:
                SimulationObj.simulation_file_path=self.sim_path+self.simulation_name+".hdf5"

        self.indexrange = irange
        self.cutoffrange = crange
        self.K_init = K
        self.j2000_generate = False

        #Response Generator Generation
        self.det_rsp = dict()
        os.chdir(self.trigger_folder)
        for det in det_list:
            # Create Response Generator for every detector
            self.det_rsp[det] = drm.DRMGenTTE(tte_file=glob('glg_tte_'+det+'_bn'+self.trigger+'_v0*.fit.gz')[0],
                                              trigdat=self.trigfile,
                                              mat_type=2,
                                              cspecfile=glob('glg_cspec_'+det+'_bn'+self.trigger+'_v0*.pha')[0],
                                              occult=False)

        os.chdir(self.directory)

        # Grid Generation
        if algorithm == 'Fibonacci':
            self.grid = self.fibonacci_sphere()
            self.generate_j2000()
            #TODO Correct the True Variable. Is it still useful?
            if self.j2000_generate is True:
                self.generate_j2000()

        for point in self.grid:
            # Generate Astromodels Spectra
            point.generate_astromodels_spectrum(i_min=float(min(irange)),
                                                i_max=float(max(irange)),
                                                c_min=float(min(crange)),
                                                c_max=float(max(crange)))
    def fibonacci_sphere(self, randomize=False):
        """
        The standard algorithm for isotropic point distribution on a sphere based on the fibonacci-series
        """
        samples = self.N
        rnd = 1.
        if randomize:
            rnd = random.random()*samples
        else: rnd=0

        points = []
        offset = 2./samples
        increment = math.pi*(3.-math.sqrt(5.))

        for i in range(samples):
            y = ((i*offset)-1)+(offset/2)
            r = math.sqrt(1-y**2)
            phi = ((i+rnd) % samples)*increment
            x = math.cos(phi)*r
            z = math.sin(phi)*r
            if rank==0:
                points.append(
                    GridPoint('gp'+str(i), [x, y, z], self.spectrum_dimension, self.det_rsp, self.K_init))
            else:
                points.append(
                    GridPoint('gp'+str(i), [x, y, z], self.spectrum_dimension, self.det_rsp,self.K_init))
        return points

    def voronoi_sphere(self):
        """
        !!BROKEN!!
        Should have been a modified better version of the fibonacci algorithm
        """
        samples = self.N
        points = []
        gr = (1+np.sqrt(5))/2
        e = 11/2
        # Sequence on [0,1]^2
        for i in range(samples-2):
            if i == 0:
                t1 = 0
                t2 = 0
            elif i == samples-2:
                t1 = 1
                t2 = 0

            else:
                t1 = ((i+e+0.5)/(samples+2*e))
                t2 = (i/gr)

            # Spherical area conserving projection
            p1 = np.arccos(2 * t1-1)-np.pi/2
            p2 = 2*np.pi * t2

            # Transformation to cartesian
            x = np.cos(p1)*np.cos(p2)
            y = np.cos(p1)*np.sin(p2)
            z = np.sin(p1)

            points.append(
                GridPoint('gp'+str(i), [x, y, z], self.spectrum_dimension,self.det_rsp,self.K_init))

        return points

    def coulomb_refining(self, Nsteps, dt=0.1):
        '''
        Refine your 'Fibonacci Lattice' with a coulomb (inverse square) repulsion
        simulation to get a physically correct isotropic distribution
        ODE solver is velocity verlet

        Parameters:
        Nsteps: Number of simulation steps
        dt: stepsize
        '''
        
        # get array of coordinates from GridPoint array particles = self.get_coords_from_gridpoints()
        particles = self.get_coords_from_gridpoints()
        if len(particles) == 1:
            print("Only one GridPoint in Grid. No Coulomb Interaction possible. No Coordinates updated!")
            return 0

        if rank==0:
            print("\nCoulomb Refining\n")
            steps = range(Nsteps)

            def _force_law(pos1, pos2):
                """force_law
                   A simple inverse squared force law
                :param pos1:
                :param pos2:
                """
                dist = np.linalg.norm(pos1-pos2, 2)
                return (pos1-pos2)/(dist**3)

            oldparticles = particles
            oldparticles_temp = particles
            it = 0
            with tml.io.progress_bar.progress_bar(Nsteps) as progress:
                for step in steps:
                    i = 0
                    for particle in particles:
                        distlist = []
                        otherparticles = np.delete(particles, i, axis=0)
                        force = np.zeros(3)

                        for otherparticle in otherparticles:
                            # Adding all N-1 particle forces

                            distance = np.linalg.norm(particle-otherparticle)
                            distlist.append(distance)

                            force = force+_force_law(particle, otherparticle)

                        normalcomponent = particle/np.linalg.norm(particle)
                        # Use only tangential part of force
                        force = force-np.dot(normalcomponent, force)*normalcomponent


                        if np.amin(distlist) > 1:
                            # If particle is very far from its closest neighbour we want to increase to force to
                            # speed up convergence to stable configuration
                            force *= 3

                        if it == 0:
                            # Velocity Verlet Initialization
                            newp = particle+0.5*force*dt**2
                            particles[i] = newp/np.linalg.norm(newp, 2)
                        else:
                            # Velocity verlet step
                            oldparticles_temp[i] = particles[i]
                            newp = 2*particle-oldparticles[i]+force*dt**2
                        particles[i] = newp/np.linalg.norm(newp, 2)
                        oldparticles[i] = oldparticles_temp[i]
                        i += 1
                    it += 1
                    progress.animate(it)

        MPI.COMM_WORLD.Barrier()
        # MPI Broadcast Grid to all processes
        particles = MPI.COMM_WORLD.bcast(particles, root=0)

        for i, gp in enumerate(self.grid):
            # Update coordinates
            gp.update_coord(particles[i])




    def generate_j2000(self, time=0.):
        '''
        Calculate Ra and Dec Values for your GridPoints
        '''
        position_interpolator = PositionInterpolator(trigdat=self.trigfile)
        self.sat_coord = position_interpolator.sc_pos(time)
        self.sat_quat = position_interpolator.quaternion(time)
        self.get_detector_cartesian()
        for gp in self.grid:
            gp.add_j2000(self.sat_coord, self.sat_quat)
            if rank==0:
                if SimulationObj.skeleton is False:
                    simulation_file = h5py.File(self.simulation_file_path,"r+")
                    if not self.j2000_generate :
                        simulation_file["grid"].create_group(gp.name)
                    simulation_file["grid"][gp.name].attrs["True Position"]=[gp.ra,gp.dec]
                    simulation_file["grid"][gp.name].attrs["True Position SC"]=gp.coord
                    simulation_file.close()

        self.j2000_generate = True
        #  except:
            #  print("Error! Is trigdat path correct?")

    def print_earth_points(self):
        gbm=GBM(self.sat_quat ,self.sat_coord*u.km)
        res=gbm.get_earth_points()
        return res

    def grid_plot(self):
        '''
        Visualize Grid
        '''
        ralist = []
        declist = []
        for point in self.grid:
            ralist.append(point.j2000.ra)
            declist.append(point.j2000.dec)
            icrsdata = coord.SkyCoord(ra=ralist*u.degree, dec=declist*u.degree, frame=coord.ICRS)
        plt.subplot(111, projection='aitoff')
        plt.grid(True)
        plt.scatter(icrsdata.ra.wrap_at('180d').radian, icrsdata.dec.radian)

    def get_coords_from_gridpoints(self):
        '''
        Returns python list of all x,y,z coordinates of the points
        '''
        pointlist = []
        for point in self.grid:
            pointlist.append(point.coord)
        return pointlist

    def generate_TRIG_spectrum(self, trigger="191017391"):
        '''
        Generates DRMs for all GridPoints for all detectors and
        folds the given spectra matrices
        through it so that we get a simulated physical photon count spectrum

        It uses sample TRIGDAT file

        Generates for every GridPoint:
        response
        response_generator

        :param trigger:

        '''

        det_list = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5',
                    'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']
        self.det_rsp = dict()
        os.chdir(self.trigger_folder)
        for det in det_list:
            rsp = drm.drmgen_trig.DRMGenTrig(
                self.sat_quat, self.sat_coord, det_list.index(det), tstart=0., tstop=2., time=0.)

            self.det_rsp[det] = rsp

        for gp in self.grid:
            ra, dec = gp.ra, gp.dec
            for det in det_list:
                gp.response[det] = self.det_rsp[det].to_3ML_response(ra, dec)
                gp.response_generator[det] = np.empty(
                    gp.dim, dtype=classmethod)
                i = 0
                for i in range(np.shape(gp.spectrum_matrix)[0]):
                    j = 0
                    for j in range(np.shape(gp.spectrum_matrix)[0]):
                        gp.response_generator[det][i, j] = tml.DispersionSpectrumLike.from_function(
                            det, source_function=gp.spectrum_matrix[i, j], background_function=self.background, response=gp.response[det],verbose = False)
        os.chdir(self.directory)


    def grid_generate_DRM_spectrum(self, snr=20., e=1.):
        """grid_generate_DRM_spectrum

        :param snr: Signal to Noise ratio
        :param e: Accepted error

        Generates DRMs for all GridPoints for all detectors and folds the given spectra matrices
        through it so that we get a simulated physical photon count spectrum

        It uses sample TTE,TRIGDAT and CSPEC files to generate the DRMs

        Generates for every GridPoint:
        response: type(InstrumentResponse)
        response_generator: type(tml.DispersionSpectrumLike)

        Automatically saves your Spectra as PHA files

        """

        val_mat=[]
        for i, gp in enumerate(self.grid):
            # Parallel DRM Generation
            if i % size != rank: continue
            print("GridPoint %d being done by processor %d \n" %(i,rank))
            gp.generate_DRM_spectrum(snr=snr)
            print("Save PHA "+str(i)+"\n")
            val_mat.append(gp.value_matrix)
            val_mat=np.array(val_mat)
            # Save PHA RSP files
            gp.save_pha(overwrite=True)

        if len(self.grid)<size and rank>=len(self.grid):
            val_mat=np.empty(self.grid[0].value_matrix.shape)

        MPI.COMM_WORLD.Barrier()
        new_val_mat=np.array(MPI.COMM_WORLD.gather(val_mat,root=0),dtype='object')
        MPI.COMM_WORLD.Barrier()
        if rank==0:
            new_val_mat=new_val_mat[:self.N]
            new_val_mat_resh=new_val_mat.reshape((self.N,self.spectrum_dimension[0],self.spectrum_dimension[1]),order='F')

        if rank == 0 and SimulationObj.skeleton is False:
            # hdf5 logging

            print("Started Logging")
            simulation_file = h5py.File(self.simulation_file_path,"r+")
            for i,gp in enumerate(self.grid):
                simulation_file['grid/'+gp.name+"/Spectrum Parameters"][...]=new_val_mat_resh[i]

                #simulation_file['grid/'+gp.name+"/Spectrum Parameters"][...]=gp.value_matrixVhk
            simulation_file.close()

    def save_DRM_spectra(self,overwrite=True):
        """save_DRM_spectra

        :param overwrite: Overwrite existing spectra
        """

        for gp in self.grid:
            gp.save_pha(overwrite)

    def load_DRM_spectra(self):
        '''
        Load saved PHA files from folder saved_pha in Simulation grid
        '''
        dirs = 0
        dirlist = []
        for _, dirnames, filenames in os.walk(self.sim_path):
            dirlist.append(dirnames)

        #Flattening dirlist
        dirlist = [item for sublist in dirlist for item in sublist]

        for item in dirlist:
            if item.startswith('gp'):
                dirs += 1

        assert len(self.grid) == dirs, "Number of gridpoints do not coincide"

        os.chdir(self.sim_path)

        for gp in self.grid:
            i_list = []
            j_list = []

            for _, dirnames, filenames in os.walk(self.sim_path+gp.name+"/PHAFiles/"):
                for filename in filenames:
                    i_list.append(filename.split("_")[1])
                    j_list.append(filename.split("_")[2][0])

            assert int(max(i_list)) == self.spectrum_dimension[0]-1 and int(
                max(j_list)) == self.spectrum_dimension[1]-1, "Dimensions do not coincide"

            for det in det_list:
                gp.response_generator[det] = np.empty(
                    gp.dim, dtype=dict)
                i = 0
                for i in range(gp.dim[0]):
                    j = 0
                    for j in range(gp.dim[1]):
                        file_name = det+"_"+str(i)+"_"+str(j)
                        file_path = gp.name + "/PHAFiles/" + file_name

                        gp.response_generator[det][i, j] = tml.OGIPLike(gp.name+"_"+file_name,
                                                                        observation = file_path+".pha",
                                                                        background = file_path+"_bak.pha",
                                                                        response = file_path+".rsp",
                                                                        spectrum_number = 1,
                                                                        verbose = False)
        os.chdir(self.directory)


    def run(self,n_detectors=4,fixed_detectors=None):
        '''
        n_detectors: number of strongest detectors to use for fitting
        '''

        for gp in self.grid:
            gp.refit_spectra(n_detectors=n_detectors,fixed_detectors=fixed_detectors)

    def run_fisher(self,n_detectors, n_samples, k,fixed_detectors):
        '''
        n_detectors: number of strongest detectors to use for fitting
        n_samples: Number of fisher samples
        k: Fisher concentration constant
        '''
        for gp in self.grid:
            if rank==0:
                simulation_file = h5py.File(self.simulation_file_path,"r+")
                simulation_file["grid/"+gp.name].create_group("fisher")
                simulation_file.close()
            gp.create_fisher_samples(k, n_samples)
            gp.refit_spectra(n_detectors=n_detectors,use_fisher_samples=True,fixed_detectors=fixed_detectors)

    def get_detector_cartesian(self):
        gbm = GBM(self.sat_quat,sc_pos=self.sat_coord*u.km)
        det=gbm.detectors
        SimulationObj.det_cartesian_dict={}
        for key in det.keys():
            _a=det[key].get_center()
            _a.representation='cartesian'
            SimulationObj.det_cartesian_dict[key] = [_a.SCX,_a.SCY,_a.SCZ]


class GridPoint(SimulationObj):

    '''A point in the simulation grid.

    Containing:
    Astromodel Spectra
    Position Information
    DispersionSpectrumLike
    Fisher Distribution
    '''

    def __init__(self, name, coord, dim, det_rsp, K_init):
        self.name = name # string "gp0"
        self.coord = coord # array [x,y,z] Coordinates
        self.dim = dim # Spectrum Matrix Dimensions tuple (2,2)
        self.j2000 = None # SkyCoord Object
        self.response = dict() # InstrumentResponse
        self.response_generator = dict() # tml.DispersionSpectrumLike
        self.det_rsp = det_rsp # DRMGen
        self.K_init=K_init # float

    def generate_astromodels_spectrum(self, i_max, i_min, c_max, c_min):
        """
        Compute sample cutoff powerlaw spectra
        spectrum_matrix:

        """
        n = self.dim[0]
        m = self.dim[1]

        self.spectrum_matrix = np.empty(self.dim, dtype=classmethod)

        self.value_matrix = np.empty(
            self.dim, dtype=[('F', 'f8'), ('xp', 'f8'), ('alpha', 'f8')])
        '''
        Array with dimension self.dim
        Contains Information about the spectral parameters of each spectrum from spectrum_matrix
        Each cell has structure [F,xp,alpha]
        '''
        self.K_matrix=np.full(self.dim,self.K_init)
        index = np.linspace(i_min, i_max, n)
        cutoff = np.linspace(c_min, c_max, m)
        if rank==0 and SimulationObj.skeleton is False:
            simulation_file = h5py.File(self.simulation_file_path,"r+")
            f=simulation_file['grid']
        i = 0
        # source=tml.PointSource()
        for index_i in index:
            j = 0
            for cutoff_i in cutoff:
                self.spectrum_matrix[i, j] = tml.Band_Calderone(F=self.K_init,alpha=index_i,xp=cutoff_i,opt=0)
                self.value_matrix[i, j]["F"] = self.K_init
                self.value_matrix[i, j]["xp"] = cutoff_i
                self.value_matrix[i, j]["alpha"] = index_i
                j += 1
            i += 1

        if rank==0 and SimulationObj.skeleton is False:
            f[self.name].create_dataset("Spectrum Parameters",self.dim,data=self.value_matrix)
            f[self.name].attrs["Spectrum Type"]="tml.Band_Calderone"
            simulation_file.close()

    def add_j2000(self, sat_coord, sat_quat, time=0.):
        """Add ra and dec coordinates to GridPoint
        Calculate the corresponding Ra and Dec coordinates
        for the already given coordinate in the Fermi-Frame

        """

        self.sat_coord=sat_coord
        self.sat_quat=sat_quat
        x, y, z = sat_coord
        q1, q2, q3, q4 = sat_quat

        self.frame = GBMFrame(sc_pos_X=x, sc_pos_Y=y, sc_pos_Z=z, quaternion_1=q1, quaternion_2=q2, quaternion_3=q3, quaternion_4=q4,
                        SCX=self.coord[0]*u.km, SCY=self.coord[1]*u.km, SCZ=self.coord[2]*u.km, representation='cartesian')
        icrsdata = gbm_frame.gbm_to_j2000(self.frame, coord.ICRS)
        self.j2000 = icrsdata
        self.ra = self.j2000.ra.degree
        self.dec = self.j2000.dec.degree

        


    def calc_j2000(self,coordinates):
        '''Rather use add_j2000 for grid simulation purposes! Calculate RA and DEC


        Returns:
        A SkyCoord object in ICRS Frame

        '''
        x, y, z = self.sat_coord
        q1, q2, q3, q4 = self.sat_quat

        frame = GBMFrame(sc_pos_X=x, sc_pos_Y=y, sc_pos_Z=z, quaternion_1=q1, quaternion_2=q2, quaternion_3=q3, quaternion_4=q4,
                        SCX=coordinates[0]*u.km, SCY=coordinates[1]*u.km, SCZ=coordinates[2]*u.km, representation='cartesian')
        return gbm_frame.gbm_to_j2000(frame, coord.ICRS)

    def show(self):
        '''
        Returns coordinates in cartesian and ICRS and a table with the generated sample spectrum parameters
        '''
        print("GBM Cartesian Coordinates: " + str(self.coord))
        if self.j2000 is None:
            print(
                "RA and DEC not calculated yet! Use generate_j2000 function of Simulator to do so.")
        else:
            print("RA: " + str(self.j2000.ra) +
                  " \nDEC: " + str(self.j2000.dec))
            display(HTML(tabulate(self.value_matrix_string, tablefmt='html',
                                  headers=range(self.dim[1]), showindex='always')))

    def save_pha(self, overwrite):
        dirpath = self.sim_path+self.name+"/PHAFiles/"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        elif overwrite:
            files=glob(dirpath+"/*")
            for f in files:
                os.remove(f)
        else:
            print("Path already exists and overwrite option is false")

        for det in det_list:
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    self.response_generator[det][i, j].write_pha(
                        dirpath+"/"+det+"_"+str(i)+"_"+str(j), overwrite=True)

        os.chdir(self.directory)

    def update_coord(self, new_coord):
        '''
        Update the coordinate of the GridPoint
        '''
        self.coord = new_coord

    def generate_DRM_spectrum(self, ra=None, dec=None, only_response=False,snr=20.,e=0.1):
        '''Generate a DispersionSpectrum with a response matrix
        Parameters


        only_response (Boolean): generate only InstrumenResponse no DispersionSpectrumLike

        Generates:
        response
        response_generator
        '''

        if ra is None and dec is None:
            ra = self.ra
            dec = self.dec

        if not only_response:
            print(self.name+": Calc Response\n")

            for det in det_list:
                self.response[det] = self.det_rsp[det].to_3ML_response(ra, dec)


                self.response_generator[det] = np.empty(
                    self.dim, dtype=classmethod)

            print(self.name+": Iterating Spectra\n")

            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    res = self.iterate_signal_to_noise(i,j,e=e,snr=snr)
                    if res=="ConvergenceError":
                        with open("ConvergenceError.csv","a") as f:
                            writer=csv.writer(f)
                            writer.writerow([self.ra,self.dec])
        else:
            response_list={}

            for det in det_list:
                response_list[det] = self.det_rsp[det].to_3ML_response(ra, dec)
            return response_list


                #for det in det_list:
                    #self.response_generator[det][i,j].update({"significance":self.response_generator[det][i, j]["generator"].significance})


    def iterate_signal_to_noise(self, i, j, snr=20.,e=0.1):
        '''
        Iterate over astromodel spectrum smplitude "F" until each gridpoint has the same signal to noise ratio with an allowed deviation of e
        '''

        bgk_K=10

        sigmax=self.calc_sig_max(bgk_K, i, j)
        sr=abs((sigmax/snr)-1)

        while sr > e:
            K_temp=self.K_matrix[i,j]*snr/sigmax

            if K_temp>1000:
                print("ConvergenceError")
            elif K_temp<1E-30:
                self.K_matrix[i,j]=np.random.randint(1,20)*1E-6
            else:
                self.K_matrix[i,j]=K_temp

            self.spectrum_matrix[i, j] = tml.Band_Calderone(F=self.K_matrix[i,j],xp=self.value_matrix[i,j]["xp"],alpha=self.value_matrix[i,j]["alpha"],opt=0)
            sigmax=self.calc_sig_max(bgk_K, i, j)
            sr=abs((sigmax/snr)-1)

        self.value_matrix[i,j]["F"]=self.K_matrix[i,j]


    def calc_sig_max(self, bgk_K, i, j):
        '''Calculate the maximum significance for GridPoint
        '''

        siglist=[]

        for det in det_list:
            self.response_generator[det][i, j] = tml.DispersionSpectrumLike.from_function(det+str(i)+str(j)+self.name, source_function=self.spectrum_matrix[i, j], background_function=tml.Powerlaw(K=bgk_K,piv=100), response=self.response[det])

            if det != 'b0' and det != 'b1':
                self.response_generator[det][i,j].set_active_measurements('8.1-900')
            else:
                self.response_generator[det][i,j].set_active_measurements('250-30000')

            siglist.append(self.response_generator[det][i,j].significance)
        return max(siglist)


    def create_fisher_samples(self, k, n_samples):
        '''create list with coordinates from fisher-bingham distribution'''
        self.fisher_samples=fb83(k*np.array(self.coord),[0,0,0]).rvs(n_samples)
        self.fisher_samples_radec=self.calc_j2000(self.fisher_samples.T)
        self.n_fisher_samples = n_samples

        if rank==0:

            simulation_file = h5py.File(self.simulation_file_path,"r+")

            for i,sample in enumerate(self.fisher_samples_radec):
                simulation_file["grid/"+self.name+"/fisher"].create_group("f"+str(i))
                simulation_file["grid/"+self.name+"/fisher/f"+str(i)].attrs["Position"]=[sample.ra.degree,sample.dec.degree]

            simulation_file.close()


    def refit_spectra(self, n_detectors=4,fixed_detectors=None,ra=None, dec=None, use_fisher_samples=False):
        '''
        Run a bayesian analysis on all Grid spectra
        use_fisher_samples:
        Run bayesian analysis for the random distributed fisher samples around original position.
        '''
        if ra is None and dec is None:
            ra = self.ra
            dec = self.dec

        if use_fisher_samples:
            self.fit_fisher_samples(n_detectors,fixed_detectors)

        else:
            self.fit_true_samples(n_detectors,fixed_detectors)

    def save_results(self,i,j,bayesian_analysis,ls,significance_dict,fisher,n=None):

        # Function should only be called in one thread at a time
        # bayesian_analysis corresponds to ba in refit_spectra
        # significance_dict should be full_sig from refit_spectra

        print("==============")
        print("Saving Results")
        print("==============")
        ij_key=(str(i),str(j))
        dirpath = self.sim_path+self.name+"/"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        if fisher:
            bayesian_analysis[ij_key].results.write_to(self.sim_path+self.name+'/Fisher/results_'+self.name+"_fisher"+str(n)+"_"+str(i)+"_"+str(j)+".fits",overwrite=True)

            simulation_file = h5py.File(self.simulation_file_path,"r+")
            fits=simulation_file["grid/"+self.name+"/fisher/f"+str(n)].create_group("("+str(i)+","+str(j)+")")
            fits.attrs["FITSPath"]=self.sim_path+self.name+'/Fisher/results_'+self.name+"_fisher"+str(n)+"_"+str(i)+"_"+str(j)+".fits"
            fits.attrs["SelectedDetectors"]=ls
            fits.attrs["SignificanceDict"]=json.dumps(significance_dict[ij_key])
            simulation_file.close()
        else:
            bayesian_analysis[ij_key].results.write_to(self.sim_path+self.name+'/results_'+self.name+"_"+str(i)+"_"+str(j)+".fits",overwrite=True)

            simulation_file = h5py.File(self.simulation_file_path,"r+")
            fits=simulation_file["grid/"+self.name].create_group("("+str(i)+","+str(j)+")")
            fits.attrs["FITSPath"]=self.sim_path+self.name+'/results_'+self.name+"_"+str(i)+"_"+str(j)+".fits"
            fits.attrs["SelectedDetectors"]=ls
            fits.attrs["SignificanceDict"]=json.dumps(significance_dict[ij_key])
            simulation_file.close()

    def fit_fisher_samples(self,n_detectors,fixed_detectors=None):

        for n,sample in enumerate(self.fisher_samples_radec):

            new_response=self.generate_DRM_spectrum(sample.ra.degree,sample.dec.degree,only_response=True)

            if rank==0:
                dirpath = self.sim_path+self.name+"/Fisher/Responses"

                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)

                for det in det_list:
                    new_response[det].to_fits(dirpath+"/f"+str(n)+"_response_"+det+".fits","test","test",overwrite=True)


            MPI.COMM_WORLD.Barrier()

            obs_path =  {(str(i), str(j)): {det : self.response_generator[det][i, j]._observed_spectrum.filename for det in det_list} for (i, j), value in np.ndenumerate(self.value_matrix)}
            bak_path =  {(str(i), str(j)): {det : self.response_generator[det][i, j]._background_spectrum.filename for det in det_list} for (i, j), value in np.ndenumerate(self.value_matrix)}

            fisher_temp_response = {ij_key : {det : tml.OGIPLike(self.name+"_"+det+"_"+str(ij_key[0])+"_"+str(ij_key[1])+"_fisher"+str(n),
                                                                 observation = self.sim_path+obs_path[ij_key][det],
                                                                 background = self.sim_path+bak_path[ij_key][det],
                                                                 response = self.sim_path+self.name+"/Fisher/Responses/f"+str(n)+"_response_"+det+".fits",
                                                                 spectrum_number = 1,
                                                                 verbose = False) for det in det_list} for ij_key in bak_path}


            # Setting up Spectrum with parameter priors

            spectrum=tml.Band_Calderone(opt=0)
            spectrum.F.prior=tml.Log_uniform_prior(lower_bound=1E-20,upper_bound=100)
            spectrum.alpha.set_uninformative_prior(tml.Uniform_prior)
            spectrum.beta.fix=True
            spectrum.xp.prior=tml.Log_uniform_prior(lower_bound=1E-20, upper_bound=10000)

            ps=tml.PointSource(self.name,ra=float(sample.ra.degree),dec=float(sample.dec.degree), spectral_shape=spectrum)
            full_sig =  {(str(i), str(j)): {det : fisher_temp_response[(str(i),str(j))][det].significance for det in det_list} for (i, j), value in np.ndenumerate(self.value_matrix)}
            selected_sig = {}
            model=tml.Model(ps)
            result=dict()
            jl={}
            ba={}
            data=dict()

            for ij_key in full_sig:

                assert isinstance(ij_key, tuple), "Key of full_sig should be tuple"
                assert isinstance(ij_key[0], str) and isinstance(ij_key[1], str), "Entries of ij_key tuple should be strings"

                i=int(ij_key[0])
                j=int(ij_key[1])
                strongest_detectors=[]
                lsval=[]

                if fixed_detectors is None and n_detectors>3:
                    sorted_detectors = sorted(full_sig[ij_key].items(), key=operator.itemgetter(1)) # increasing
                    strongest_detectors= [tpl[0] for tpl in sorted_detectors[-(n_detectors+1):]] # increasing

                    del_index=0
                    # Check for rule only one bgo detector
                    if ("b0" in strongest_detectors) and ("b1" in strongest_detectors):
                        del_index+=1
                        if strongest_detectors.index("b0")>strongest_detectors.index("b1"):
                            strongest_detectors.remove("b1")
                        else:
                            strongest_detectors.remove("b0")
                    else:
                        del_index+=1
                        del strongest_detectors[0]

                    # check that detectors dont lie on a line -> bad localization
                    det = SimulationObj.det_cartesian_dict # TODO Have to set to SimulationObj
                    smallest_angles_in_sub_triangles=[]
                    for k in strongest_detectors:
                        ls_temp=copy.copy(strongest_detectors)
                        ls_temp.remove(k)
                        sub_triangle=ls_temp
                        angles_in_sub_triangles=[]
                        for sub_triangle in multiset_permutations(ls_temp):
                            angles_in_sub_triangles.append(spg.great_circle_arc.angle(det[sub_triangle[0]],det[sub_triangle[1]],det[sub_triangle[2]]))
                        smallest_angles_in_sub_triangles.append(min(angles_in_sub_triangles))

                    biggest_smallest_angle = max(smallest_angles_in_sub_triangles)

                    if 0<biggest_smallest_angle<10:
                        del_index+=1
                        del strongest_detectors[0]
                        if not sorted_detectors[-(n_detectors+del_index)] in ['b1','b2']:
                            strongest_detectors.insert(0,sorted_detectors[-(n_detectors+del_index)][0])
                        else:
                            strongest_detectors.insert(0,sorted_detectors[-(n_detectors+del_index+1)][0])





                else:
                    strongest_detectors = fixed_detectors

                ls=strongest_detectors
                selected_sig[ij_key]=strongest_detectors

                #data[ij_key]=tml.DataList(*[drm.BALROGLike.from_spectrumlike(self.response_generator[det][i,j],0,self.det_rsp[det]) for det in selected_sig[ij_key]])
                for det in ls:
                    if det != 'b0' and det != 'b1':
                        fisher_temp_response[(str(i),str(j))][det].set_active_measurements('8.1-900')
                    else:
                        fisher_temp_response[(str(i),str(j))][det].set_active_measurements('250-30000')

                data[ij_key]=tml.DataList(*[fisher_temp_response[(str(i),str(j))][det] for det in selected_sig[ij_key]])

                ba[ij_key]=tml.BayesianAnalysis(model,data[ij_key])

                if rank == 0:
                    print("========================")
                    print("Fitting fisher spectrum number "+ str(n)+"\/"+str(self.n_fisher_samples)+" of "+ self.name + " (" + str(i)+","+str(j)+")")
                    print("========================")

                ba[ij_key].sample_multinest(800,verbose=False,resume=False,importance_nested_sampling=False)

                if rank==0:
                    self.save_results(i,j,ba,ls,full_sig,True,n)

                    #  print("==============")
                    #  print("Saving Results")
                    #  print("==============")
                    #  ba[ij_key].results.write_to(self.sim_path+self.name+'/Fisher/results_'+self.name+"_fisher"+str(n)+"_"+str(i)+"_"+str(j)+".fits",overwrite=True)
                    #  simulation_file = h5py.File(self.simulation_file_path,"r+")
                    #  fits=simulation_file["grid/"+self.name+"/fisher/f"+str(n)].create_group("("+str(i)+","+str(j)+")")
                    #  fits.attrs["FITSPath"]=self.sim_path+self.name+'/Fisher/results_'+self.name+"_fisher"+str(n)+"_"+str(i)+"_"+str(j)+".fits"
                    #  fits.attrs["SelectedDetectors"]=ls
                    #  fits.attrs["SignificanceDict"]=json.dumps(full_sig[ij_key])
                    #  simulation_file.close()
            del data
            fisher_temp_response=None
            del new_response
            obs_path=None
            bak_path=None
            del full_sig
            del ba
            gc.collect()



    def fit_true_samples(self,n_detectors,fixed_detectors):
        spectrum=tml.Band_Calderone(opt=0)
        spectrum.F.prior=tml.Log_uniform_prior(lower_bound=1E-20,upper_bound=100)
        spectrum.alpha.set_uninformative_prior(tml.Uniform_prior)
        spectrum.beta.fix=True
        spectrum.xp.prior=tml.Log_uniform_prior(lower_bound=1E-20, upper_bound=10000)

        ps=tml.PointSource(self.name,ra=self.ra,dec=self.dec, spectral_shape=spectrum)
        model=tml.Model(ps)

        full_sig =  {(str(i), str(j)): {det : self.response_generator[det][i, j].significance for det in det_list} for (i, j), value in np.ndenumerate(self.value_matrix)}
        selected_sig = {}
        result=dict()
        jl={}
        ba={}
        data=dict()

        for ij_key in full_sig:

            assert isinstance(ij_key, tuple), "Key of full_sig should be tuple"
            assert isinstance(ij_key[0], str) and isinstance(ij_key[1], str), "Entries of ij_key tuple should be strings"

            i=int(ij_key[0])
            j=int(ij_key[1])
            strongest_detectors=[]
            lsval=[]

            if fixed_detectors is None and n_detectors>3:
                sorted_detectors = sorted(full_sig[ij_key].items(), key=operator.itemgetter(1)) # increasing
                strongest_detectors= [tpl[0] for tpl in sorted_detectors[-(n_detectors+1):]] # increasing

                del_index=0
                # Check for rule only one bgo detector
                if ("b0" in strongest_detectors) and ("b1" in strongest_detectors):
                    del_index+=1
                    if strongest_detectors.index("b0")>strongest_detectors.index("b1"):
                        strongest_detectors.remove("b1")
                    else:
                        strongest_detectors.remove("b0")
                else:
                    del_index+=1
                    del strongest_detectors[0]

                # check that detectors dont lie on a line -> bad localization
                det = SimulationObj.det_cartesian_dict # TODO Have to set to SimulationObj
                smallest_angles_in_sub_triangles=[]
                for k in strongest_detectors:
                    ls_temp=copy.copy(strongest_detectors)
                    ls_temp.remove(k)
                    sub_triangle=ls_temp
                    angles_in_sub_triangles=[]
                    for sub_triangle in multiset_permutations(ls_temp):
                        angles_in_sub_triangles.append(spg.great_circle_arc.angle(det[sub_triangle[0]],det[sub_triangle[1]],det[sub_triangle[2]]))
                    smallest_angles_in_sub_triangles.append(min(angles_in_sub_triangles))

                biggest_smallest_angle = max(smallest_angles_in_sub_triangles)

                if 0<biggest_smallest_angle<10:
                    del_index+=1
                    del strongest_detectors[0]
                    if not sorted_detectors[-(n_detectors+del_index)] in ['b1','b2']:
                        strongest_detectors.insert(0,sorted_detectors[-(n_detectors+del_index)][0])
                    else:
                        strongest_detectors.insert(0,sorted_detectors[-(n_detectors+del_index+1)][0])





            else:
                strongest_detectors = fixed_detectors

            ls=strongest_detectors
            selected_sig[ij_key]=strongest_detectors

            for det in ls:
                if det != 'b0' and det != 'b1':
                    self.response_generator[det][i,j].set_active_measurements('8.1-900')
                else:
                    self.response_generator[det][i,j].set_active_measurements('250-30000')

            data[ij_key]=tml.DataList(*[self.response_generator[det][i,j] for det in selected_sig[ij_key]])
            ba[ij_key]=tml.BayesianAnalysis(model,data[ij_key])
            if rank == 0:
                print("========================")
                print("Fitting true spectrum of "+ self.name + " (" + str(i)+","+str(j)+")")
                print("========================")
            ba[ij_key].sample_multinest(1000,
                                        verbose=False,
                                        resume=False,
                                        importance_nested_sampling=False)

            if rank==0:

                self.save_results(i,j,ba,ls,full_sig,False)
                #  print("==============")
                #  print("Saving Results")
                #  print("==============")
                #  dirpath = self.sim_path+self.name+"/"
                #  if not os.path.exists(dirpath):
                #      os.makedirs(dirpath)
                #  ba[ij_key].results.write_to(self.sim_path+self.name+'/results_'+self.name+"_"+str(i)+"_"+str(j)+".fits",overwrite=True)
                #
                #  simulation_file = h5py.File(self.simulation_file_path,"r+")
                #  fits=simulation_file["grid/"+self.name].create_group("("+str(i)+","+str(j)+")")
                #  fits.attrs["FITSPath"]=self.sim_path+self.name+'/results_'+self.name+"_"+str(i)+"_"+str(j)+".fits"
                #  fits.attrs["SelectedDetectors"]=ls
                #  fits.attrs["SignificanceDict"]=json.dumps(full_sig[ij_key])
                #  simulation_file.close()
        del data
        del full_sig
        del ba
        gc.collect()

