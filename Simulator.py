from tabulate import tabulate
import operator
from IPython.display import HTML, display
from gbmgeometry import PositionInterpolator, GBMFrame, gbm_frame, GBM
import astropy.coordinates as coord
import astropy.units as u
from scipy.stats import f as fisher_f
from threeML.utils.OGIP.response import OGIPResponse
from threeML import *
import astromodels
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
import csv
import math
import os
from glob import glob
import gbm_drm_gen as drm
from sphere.distribution import fb83
import h5py
mpl.use('Agg')
from mpi4py import MPI
rank=MPI.COMM_WORLD.Get_rank()
size=MPI.COMM_WORLD.Get_size()


class Simulator():

    """
    Fermi GBM Simulator

    """
    global det_list
    det_list = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5',
                'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

    def __init__(self, name, source_number, spectrum_matrix_dimensions, directory, trigger = "131229277"):
        self.simulation_name=name
        self.N = source_number
        self.spectrum_dimension = spectrum_matrix_dimensions
        self.grid = None
        self.j2000_generate = False
        self.indexrange = None
        self.cutoffrange = None
        self.directory= directory
        self.trigger = trigger
        self.trigger_folder= self.directory +"/rawdata/"+trigger
        self.trigfile = glob(self.trigger_folder + '/glg_trigdat_all_bn'+trigger+'_v0*.fit')[0]

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
                    GridPoint('gp'+str(i), [x, y, z], self.spectrum_dimension, self.det_rsp,self.K_init,self.simulation_file))
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
        print(particles)
        if len(particles)==1:
            print("Only one GridPoint in Grid. No Coulomb Interaction possible. No Coordinates updated!")
            return 0
        if rank==0: 
            
            print(particles)

            steps = range(Nsteps)
            
            def force_law(pos1, pos2):
                dist = np.linalg.norm(pos1-pos2, 2)
                return (pos1-pos2)/(dist**3)

            oldparticles = particles
            oldparticles_temp = particles
            it = 0
            for step in steps:
                i = 0
                for particle in particles:
                    distlist = []
                    otherparticles = np.delete(particles, i, axis=0)
                    force = np.zeros(3)

                    for otherparticle in otherparticles:

                        distance = np.linalg.norm(particle-otherparticle)
                        distlist.append(distance)

                        force = force+force_law(particle, otherparticle)

                    normalcomponent = particle/np.linalg.norm(particle)
                    force = force-np.dot(normalcomponent, force)*normalcomponent

                    if np.amin(distlist) > 1:
                        force *= 3

                    if it == 0:
                        newp = particle+0.5*force*dt**2
                        particles[i] = newp/np.linalg.norm(newp, 2)
                    else:
                        oldparticles_temp[i] = particles[i]
                        newp = 2*particle-oldparticles[i]+force*dt**2
                    particles[i] = newp/np.linalg.norm(newp, 2)
                    oldparticles[i] = oldparticles_temp[i]
                    i += 1
                it += 1

        MPI.COMM_WORLD.Barrier()
        particles = MPI.COMM_WORLD.bcast(particles,root=0)
        for i,gp in enumerate(self.grid):
            gp.update_coord(particles[i])

    def setup(self,K, irange=[-2, -1], crange=[100, 400], algorithm='Fibonacci' ):
        '''
        Setup the GRB grid the spectrum matrices
        and the background function for your Simulation
        '''
        if rank==0:
            self.simulation_file_path=self.directory+"/SimulationFiles/"+self.simulation_name+".hdf5"
            self.simulation_file=h5py.File(self.simulation_file_path,"w")
            f=self.simulation_file
            gridh5=f.create_group("grid")

        self.indexrange = irange
        self.cutoffrange = crange
        self.K_init = K
        self.j2000_generate=False

        #Response Generator Generation
        self.det_rsp = dict()
        os.chdir(self.trigger_folder)
        for det in det_list:
            rsp = drm.DRMGenTTE(tte_file=glob('glg_tte_'+det+'_bn'+self.trigger+'_v0*.fit.gz')[0], trigdat=self.trigfile, mat_type=2, cspecfile=glob('glg_cspec_'+det+'_bn'+self.trigger+'_v0*.pha')[0])

            self.det_rsp[det] = rsp
        os.chdir(self.directory)

        # Grid Generation
        if algorithm == 'Fibonacci':
            self.grid = self.fibonacci_sphere()
            self.generate_j2000()
            print(self.grid[0].ra)
            #TODO Correct the True Variable. Is it still useful? 
            if self.j2000_generate == True:
                self.generate_j2000()

        for point in self.grid:
            point.generate_astromodels_spectrum(i_min=float(min(irange)), i_max=float(max(irange)), c_min=float(min(crange)), c_max=float(max(crange)))



    def generate_j2000(self, time=0.):
        '''
        Calculate Ra and Dec Values for your GridPoints
        '''
        position_interpolator = PositionInterpolator(trigdat=self.trigfile)
        self.sat_coord = position_interpolator.sc_pos(time)
        self.sat_quat = position_interpolator.quaternion(time)
        try:
            for gp in self.grid:
                gp.add_j2000(self.sat_coord, self.sat_quat)
                if rank==0:
                    if not self.j2000_generate :
                        self.simulation_file["grid"].create_group(gp.name)
                    self.simulation_file["grid"][gp.name].attrs["True Position"]=[gp.ra,gp.dec]

            self.j2000_generate = True
        except:
            print("Error! Is trigdat path correct?")

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
        icrsdata = coord.SkyCoord(
            ra=ralist*u.degree, dec=declist*u.degree, frame=coord.ICRS)
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
                        gp.response_generator[det][i, j] = DispersionSpectrumLike.from_function(
                            det, source_function=gp.spectrum_matrix[i, j], background_function=self.background, response=gp.response[det])
        os.chdir(self.directory)


    def grid_generate_DRM_spectrum(self, snr=20., e=1.):
        '''
        Generates DRMs for all GridPoints for all detectors and folds the given spectra matrices
        through it so that we get a simulated physical photon count spectrum

        It uses sample TTE,TRIGDAT and CSPEC files to generate the DRMs

        Generates for every GridPoint:
        response: type(InstrumentResponse)
        response_generator: type(DispersionSpectrumLike)

        save: Save your response_generator as PHA file in the folder saved_pha for MPI4PY data distribution

        '''

        for i,gp in enumerate(self.grid):
            if i%size!=rank: continue
            print("GridPoint %d being done by processor %d" %(i,rank))
            gp.generate_DRM_spectrum()
            print("Save PHA "+str(i))
            gp.save_pha(self.directory,overwrite=True)

        MPI.COMM_WORLD.Barrier()

        if rank==0:
            for gp in self.grid:
                self.simulation_file['grid/'+gp.name+"/Spectrum Parameters"][...]=gp.value_matrix

    def save_DRM_spectra(self,overwrite=True):

        for gp in self.grid:
            gp.save_pha(self.directory,overwrite)

    def load_DRM_spectra(self):
        '''
        Load saved PHA files from folder saved_pha in Simulation grid
        '''
        i_list = []
        j_list = []
        dirs = 0
        for _, dirnames, filenames in os.walk(self.directory+"/SimulationFiles/PHAFiles/"):
            for filename in filenames:
                i_list.append(filename.split("_")[1])
                j_list.append(filename.split("_")[2][0])
            dirs += len(dirnames)

        assert len(self.grid) == dirs, "Number of gridpoints do not coincide"
        assert int(max(i_list)) == self.spectrum_dimension[0]-1 and int(
            max(j_list)) == self.spectrum_dimension[1]-1, "Dimensions do not coincide"

        os.chdir(self.directory+"/SimulationFiles/PHAFiles/")
        for gp in self.grid:

            for det in det_list:
                gp.response_generator[det] = np.empty(
                    gp.dim, dtype=dict)
                i = 0
                for i in range(gp.dim[0]):
                    j = 0
                    for j in range(gp.dim[1]):
                        file_name = det+"_"+str(i)+"_"+str(j)
                        file_path = gp.name + "/" + file_name

                        gp.response_generator[det][i, j] = OGIPLike(gp.name+"_"+file_name, observation=file_path+".pha", background=file_path+"_bak.pha", response=file_path+".rsp", spectrum_number=1)
        os.chdir(self.directory)

    def run(self,n_detectors=4):
        '''
        n_detectors: number of strongest detectors to use for fitting
        '''

        for gp in self.grid:
            gp.refit_spectra(self.directory,n_detectors=n_detectors)

    def run_fisher(self,n_detectors, n_samples, k):
        '''
        n_detectors: number of strongest detectors to use for fitting
        n_samples: Number of fisher samples
        k: Fisher concentration constant
        '''
        for gp in self.grid:
            if rank==0:
                self.simulation_file["grid/"+gp.name].create_group("fisher")
            gp.create_fisher_samples(k, n_samples)
            gp.refit_spectra(self.directory,n_detectors=n_detectors,use_fisher_samples=True)


class GridPoint():

    '''
    One point in the simulation grid.

    Containing:
    Astromodel Spectra
    Position Information
    DispersionSpectrumLike
    Fisher Distribution
    '''

    def __init__(self, name, coord, dim, det_rsp, K_init, simulation_file=None):
        self.name = name # string "gp0"
        self.coord = coord # array [x,y,z] Coordinates
        self.dim = dim # Spectrum Matrix Dimensions tuple (2,2)
        self.j2000 = None # SkyCoord Object
        self.response = dict() # InstrumentResponse
        self.response_generator = dict() # DispersionSpectrumLike
        self.det_rsp = det_rsp # DRMGen
        self.K_init=K_init # float
        self.simulation_file = simulation_file # h5py.File

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
        if rank==0:
            f=self.simulation_file['grid']
        i = 0
        # source=PointSource()
        for index_i in index:
            j = 0
            for cutoff_i in cutoff:
                self.spectrum_matrix[i, j] = Band_Calderone(F=self.K_init,alpha=index_i,xp=cutoff_i,opt=0)
                self.value_matrix[i, j]["F"] = self.K_init
                self.value_matrix[i, j]["xp"] = cutoff_i
                self.value_matrix[i, j]["alpha"] = index_i
                j += 1
            i += 1

        if rank==0:
            f[self.name].create_dataset("Spectrum Parameters",self.dim,data=self.value_matrix)
            f[self.name].attrs["Spectrum Type"]="Band_Calderone"

    def add_j2000(self, sat_coord, sat_quat, time=0.):
        """
        Calculate the corresponding Ra and Dec coordinates
        for the already given coordinate in the Fermi-Frame

        final_frame:
        doesnt matter as gbm_frame.gbm_to_j2000 outputs only ICRS
        """
        self.sat_coord=sat_coord
        self.sat_quat=sat_quat
        x, y, z = sat_coord
        q1, q2, q3, q4 = sat_quat

        frame = GBMFrame(sc_pos_X=x, sc_pos_Y=y, sc_pos_Z=z, quaternion_1=q1, quaternion_2=q2, quaternion_3=q3, quaternion_4=q4,
                        SCX=self.coord[0]*u.km, SCY=self.coord[1]*u.km, SCZ=self.coord[2]*u.km, representation='cartesian')
        icrsdata = gbm_frame.gbm_to_j2000(frame, coord.ICRS)
        self.j2000 = icrsdata
        self.ra = self.j2000.ra.degree
        self.dec = self.j2000.dec.degree

    def calc_j2000(self,coordinates):
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
        if self.j2000 == None:
            print(
                "RA and DEC not calculated yet! Use generate_j2000 function of Simulator to do so.")
        else:
            print("RA: " + str(self.j2000.ra) +
                  " \nDEC: " + str(self.j2000.dec))
            display(HTML(tabulate(self.value_matrix_string, tablefmt='html',
                                  headers=range(self.dim[1]), showindex='always')))

    def save_pha(self, directory, overwrite):
        dirpath = directory+"/SimulationFiles/PHAFiles/"+self.name
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

        os.chdir(directory)

    def update_coord(self, new_coord):
        '''
        Update the coordinate of the GridPoint

        '''
        self.coord = new_coord

    def generate_DRM_spectrum(self, ra=None, dec=None, only_response=False,e=0.1):
        '''
        Generate a DispersionSpectrum with a response matrix
        only_response = True:
        Generate only a 3ML InstrumentResponse for given RA and DEC

        Generates:
        response
        response_generator
        '''
        if ra==None and dec == None:
            ra = self.ra
            dec = self.dec
            print(ra)
            print(dec)
        if not only_response:
            print("Calc Response")
            for det in det_list:
                self.response[det] = self.det_rsp[det].to_3ML_response(ra, dec)

                print("Response done")
                
                self.response_generator[det] = np.empty(
                    self.dim, dtype=classmethod)
            print("Iterating")
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    res = self.iterate_signal_to_noise(i,j,e=e)
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
        Iterate over astromodel spectrum smplitude "F" so that each gridpoint has around the same signal to noise ratio
        '''

        bgk_K=10

        sigmax=self.calc_sig_max(bgk_K, i, j)
        sr=abs((sigmax/snr)-1)

        while sr > e:
            K_temp=self.K_matrix[i,j]*snr/sigmax
            # for det in det_list:
            #     self.response_generator[det][i,j]['generator'].view_count_spectrum().savefig("result_"+str(det)+"_"+str(round(bgk_K,5))+"_"+str(self.K_matrix[i,j])+".png")
            if K_temp>1000:
                print("ConvergenceError")
            elif K_temp<1E-30:
                self.K_matrix[i,j]=np.random.randint(1,20)*1E-6
            else:
                self.K_matrix[i,j]=K_temp

            self.spectrum_matrix[i, j] = Band_Calderone(F=self.K_matrix[i,j],xp=200,opt=0)
            sigmax=self.calc_sig_max(bgk_K, i, j)
            sr=abs((sigmax/snr)-1)
            print("New K: "+ str(self.K_matrix[i,j]))

        self.value_matrix[i,j]["F"]=self.K_matrix[i,j]


    def calc_sig_max(self, bgk_K, i, j):
        siglist=[]
        print(self.K_matrix)
        for det in det_list:
            self.response_generator[det][i, j] = DispersionSpectrumLike.from_function(det+str(i)+str(j)+self.name, source_function=self.spectrum_matrix[i, j], background_function=Powerlaw(K=bgk_K,piv=100), response=self.response[det])
            siglist.append(self.response_generator[det][i,j].significance)
        return max(siglist)

    def create_fisher_samples(self, k, n_samples):
        '''create list with coordinates from fisher-bingham distribution'''
        self.fisher_samples=fb83(k*np.array(self.coord),[0,0,0]).rvs(n_samples)
        self.fisher_samples_radec=self.calc_j2000(self.fisher_samples.T)
        if rank==0:
            for i,sample in enumerate(self.fisher_samples_radec):
                self.simulation_file["grid/"+self.name+"/fisher"].create_group("f"+str(i))
                self.simulation_file["grid/"+self.name+"/fisher/f"+str(i)].attrs["Position"]=[sample.ra.degree,sample.dec.degree]

    def refit_spectra(self, directory, n_detectors=4,ra=None, dec=None, use_fisher_samples=False):
        '''
        Run a bayesian analysis on all Grid spectra
        use_fisher_samples:
        Run bayesian analysis for the random distributed fisher samples around original position.
        '''
        if ra==None and dec == None:
            ra = self.ra
            dec = self.dec

        if use_fisher_samples:

            for n,sample in enumerate(self.fisher_samples_radec):

                new_response=self.generate_DRM_spectrum(sample.ra,sample.dec,only_response=True)

                # for (i,j),value in np.ndenumerate(self.value_matrix):
                #     for det in det_list:
                #         self.response_generator[det][i,j].response=new_response[det]

                spectrum=Band_Calderone(opt=0)
                spectrum.F.prior=Log_uniform_prior(lower_bound=1E-20,upper_bound=100)
                spectrum.alpha.set_uninformative_prior(Uniform_prior)
                spectrum.beta.fix=True
                spectrum.xp.prior=Log_uniform_prior(lower_bound=1E-20, upper_bound=10000)

                ps=PointSource(self.name,ra=float(sample.ra.degree),dec=float(sample.dec.degree), spectral_shape=spectrum)
                full_sig =  {(str(i), str(j)): {det : self.response_generator[det][i, j].significance for det in det_list} for (i, j), value in np.ndenumerate(self.value_matrix)}
                selected_sig = {}
                model=Model(ps)
                result=dict()
                jl={}
                ba={}
                data=dict()
                for ij_key in full_sig:
                    i=int(ij_key[0])
                    j=int(ij_key[1])
                    ls=[]
                    lsval=[]
                    for tuple in sorted(full_sig[ij_key].items(), key=operator.itemgetter(1))[-n_detectors:]:
                        ls.append(tuple[0])
                    selected_sig[ij_key]=ls
                    #data[ij_key]=DataList(*[drm.BALROGLike.from_spectrumlike(self.response_generator[det][i,j],0,self.det_rsp[det]) for det in selected_sig[ij_key]])
                    data[ij_key]=DataList(*[drm.BALROGLike.from_spectrumlike(self.response_generator[det][i,j],0.,self.det_rsp[det],free_position=False) for det in selected_sig[ij_key]])
                    # jl[ij_key]=JointLikelihood(model,data[ij_key])
                    # result[ij_key]=jl[ij_key].fit()
                    ba[ij_key]=BayesianAnalysis(model,data[ij_key])
                    ba[ij_key].sample_multinest(400,verbose=True,resume=False,importance_nested_sampling=False)
                    if rank==0:
                        dirpath = directory+"/SimulationFiles/"+self.name+"/Fisher/"
                        if not os.path.exists(dirpath):
                            os.makedirs(dirpath)
                        ba[ij_key].results.write_to(directory+'/SimulationFiles/'+self.name+'/Fisher/results_'+self.name+"_fisher"+str(n)+"_"+str(i)+"_"+str(j)+".fits",overwrite=True)
                        fits=self.simulation_file["grid/"+self.name+"/fisher/f"+str(n)].create_group("("+str(i)+","+str(j)+")")
                        fits.attrs["FITSPath"]=directory+'/SimulationFiles/'+self.name+'/Fisher/results_'+self.name+"_fisher"+str(n)+"_"+str(i)+"_"+str(j)+".fits"

        else:
            # spectrum=Cutoff_powerlaw()
                # spectrum.K.prior=Log_uniform_prior(lower_bound=1E-3,upper_bound=1000)
                # spectrum.index.set_uninformative_prior(Uniform_prior)
                # spectrum.xc.prior=Log_uniform_prior(lower_bound=1E-20,upper_bound=10000)
            spectrum=Band_Calderone(opt=0)
            spectrum.F.prior=Log_uniform_prior(lower_bound=1E-20,upper_bound=100)
            spectrum.alpha.set_uninformative_prior(Uniform_prior)
            spectrum.beta.fix=True
            spectrum.xp.prior=Log_uniform_prior(lower_bound=1E-20, upper_bound=10000)

            ps=PointSource(self.name,ra=self.ra,dec=self.dec, spectral_shape=spectrum)
            full_sig =  {(str(i), str(j)): {det : self.response_generator[det][i, j].significance for det in det_list} for (i, j), value in np.ndenumerate(self.value_matrix)}
            selected_sig = {}
            model=Model(ps)
            result=dict()
            jl={}
            ba={}
            data=dict()
            for ij_key in full_sig:
                i=int(ij_key[0])
                j=int(ij_key[1])
                ls=[]
                lsval=[]
                for tuple in sorted(full_sig[ij_key].items(), key=operator.itemgetter(1))[-n_detectors:]:
                    ls.append(tuple[0])
                    selected_sig[ij_key]=ls
                    #data[ij_key]=DataList(*[drm.BALROGLike.from_spectrumlike(self.response_generator[det][i,j],0,self.det_rsp[det]) for det in selected_sig[ij_key]])
                data[ij_key]=DataList(*[self.response_generator[det][i,j] for det in selected_sig[ij_key]])
                # jl[ij_key]=JointLikelihood(model,data[ij_key])
                # result[ij_key]=jl[ij_key].fit()
                ba[ij_key]=BayesianAnalysis(model,data[ij_key])
                ba[ij_key].sample_multinest(400,verbose=True,resume=False,importance_nested_sampling=False)
                if rank==0:
                    dirpath = directory+"/SimulationFiles/"+self.name+"/"
                    if not os.path.exists(dirpath):
                        os.makedirs(dirpath)
                    ba[ij_key].results.write_to(directory+'/SimulationFiles/'+self.name+'/results_'+self.name+"_"+str(i)+"_"+str(j)+".fits",overwrite=True)
                    fits=self.simulation_file["grid/"+self.name].create_group("("+str(i)+","+str(j)+")")
                    fits.attrs["FITSPath"]=directory+'/SimulationFiles/'+self.name+'/results_'+self.name+"_"+str(i)+"_"+str(j)+".fits"




