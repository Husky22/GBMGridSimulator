from tabulate import tabulate
from IPython.display import HTML, display
from gbmgeometry import PositionInterpolator, GBMFrame, gbm_frame
import astropy.coordinates as coord
import astropy.units as u
from threeML.utils.OGIP.response import OGIPResponse
from threeML import *
import astromodels
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
import math
import os
from glob import glob
import gbm_drm_gen as drm
mpl.use('Agg')


class Simulator():

    """
    Fermi GBM Simulator

    """
    global det_list
    det_list = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5',
                'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

    def __init__(self, source_number, spectrum_matrix_dimensions, trigfile):
        self.N = source_number
        self.spectrum_dimension = spectrum_matrix_dimensions
        self.grid = None
        self.j2000_generate = False
        self.indexrange = None
        self.cutoffrange = None
        self.trigfile = trigfile

    def fibonacci_sphere(self, randomize=True):
        """
        The standard algorithm for isotropic point distribution on a sphere based on the fibonacci-series
        """
        samples = self.N
        rnd = 1.
        if randomize:
            rnd = random.random()*samples
            points = []
            offset = 2./samples
            increment = math.pi*(3.-math.sqrt(5.))

            for i in range(samples):
                y = ((i*offset)-1)+(offset/2)
                r = math.sqrt(1-y**2)
                phi = ((i+rnd) % samples)*increment
                x = math.cos(phi)*r
                z = math.sin(phi)*r
                points.append(
                    GridPoint('gp'+str(i), [x, y, z], self.spectrum_dimension))
            return np.array(points)

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
                GridPoint('gp'+str(i), [x, y, z], self.spectrum_dimension))

        return np.array(points)

    def coulomb_refining(self, Nsteps, dt=0.1):
        '''
        Refine your 'Fibonacci Lattice' with a coulomb (inverse square) repulsion
        simulation to get a physically correct isotropic distribution
        ODE solver is velocity verlet

        Parameters:
        Nsteps: Number of simulation steps
        dt: stepsize
        '''
        # get array of coordinates from GridPoint array
        particles = self.get_coords_from_gridpoints()

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

        for i in range(len(self.grid)):
            self.grid[i].update_coord(particles[i])

    def setup(self, irange=[-1.5, -1], crange=[100, 400], K=50, algorithm='Fibonacci' ):
        '''
        Setup the GRB grid the spectrum matrices
        and the background function for your Simulation
        '''

        self.indexrange = irange
        self.cutoffrange = crange

        if algorithm == 'Fibonacci':
            self.grid = self.fibonacci_sphere()
            if self.j2000_generate == True:
                self.generate_j2000()

        elif algorithm == 'ModifiedFibonacci':
            self.grid = self.voronoi_sphere()
            if self.j2000_generate == True:
                self.generate_j2000()

        for point in self.grid:
            point.generate_spectrum(i_min=float(min(irange)), i_max=float(max(irange)), c_min=float(min(crange)), c_max=float(max(crange)), K=K)

        trigger = "131229277"  # TODO: Set variable in setup or get from Trigfile
        self.det_rsp = dict()
        os.chdir('rawdata/'+trigger)
        print(os.getcwd())
        for det in det_list:
            rsp = drm.DRMGenTTE(tte_file=glob('glg_tte_'+det+'_bn'+trigger+'_v0*.fit.gz')[0], trigdat=glob('glg_trigdat_all_bn'+trigger+'_v0*.fit')[0], mat_type=2, cspecfile=glob('glg_cspec_'+det+'_bn'+trigger+'_v0*.pha')[0])

            self.det_rsp[det] = rsp
        os.chdir("../../")

    def generate_j2000(self, time=0.):
        '''
        Calculate Ra and Dec Values for your GridPoints
        '''
        print(os.getcwd())
        position_interpolator = PositionInterpolator(trigdat=self.trigfile)
        self.sat_coord = position_interpolator.sc_pos(time)
        self.sat_quat = position_interpolator.quaternion(time)
        try:
            for gp in self.grid:
                gp.add_j2000(self.sat_coord, self.sat_quat)
            self.j2000_generate = True
        except:
            print("Error! Is trigdat path correct?")

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
        os.chdir('rawdata/'+trigger)
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
        os.chdir('../../')

    def generate_spectrum_DRMgiven(self, trigger="191017391"):
        '''
        Test for Error finding in DRM generation


        Generates for every GridPoint:
        response
        response_generator

        '''

        det_list = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5',
                    'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']
        self.det_rsp = dict()
        os.chdir('rawdata/'+trigger)
        for det in det_list:
            rsp = drm.drmgen_trig.DRMGenTrig(
                self.sat_quat, self.sat_coord, det_list.index(det), tstart=0., tstop=2., time=0.)

            self.det_rsp[det] = rsp

        for gp in self.grid:
            ra, dec = gp.ra, gp.dec
            for det in det_list:
                gp.response[det] = OGIPResponse(
                    glob('glg_cspec_'+det+'_bn'+trigger+'_v0*.rsp')[0])
                gp.response_generator[det] = np.empty(
                    gp.dim, dtype=classmethod)
                i = 0
                for i in range(gp.dim[0]):
                    j = 0
                    for j in range(gp.dim[1]):
                        gp.response_generator[det][i, j] = DispersionSpectrumLike.from_function(
                            det, source_function=gp.spectrum_matrix[i, j], background_function=self.background, response=gp.response[det])
        os.chdir('../../')

    def generate_DRM_spectrum(self, trigger="191017391", save=False, snr=20., e=1.):
        '''
        Generates DRMs for all GridPoints for all detectors and folds the given spectra matrices
        through it so that we get a simulated physical photon count spectrum

        It uses sample TTE,TRIGDAT and CSPEC files to generate the DRMs

        Generates for every GridPoint:
        response: type(InstrumentResponse)
        response_generator: type(DispersionSpectrumLike)

        save: Save your response_generator as PHA file in the folder saved_pha for MPI4PY data distribution

        '''

        for gp in self.grid:
            ra, dec = gp.ra, gp.dec
            i = 0
            for det in det_list:
                gp.response[det] = self.det_rsp[det].to_3ML_response(ra, dec)
                gp.response_generator[det] = np.empty(
                    gp.dim, dtype=dict)
            for i in range(gp.dim[0]):
                j = 0
                for j in range(gp.dim[1]):
                    self.iterate_signal_to_noise(gp,i,j)
                    for det in det_list:
                        gp.response_generator[det][i,j].update({"significance":gp.response_generator[det][i, j]["generator"].significance})
                        if save == True:
                            dirpath = "saved_pha/"+gp.name
                            if not os.path.exists(dirpath):
                                os.makedirs(dirpath)
                            gp.response_generator[det][i, j]["generator"].write_pha(
                                dirpath+"/"+det+"_"+str(i)+"_"+str(j), overwrite=True)

        os.chdir('../../')

    def iterate_signal_to_noise(self, gp, i, j, snr=20.):

        e=0.01
        bgk_K=20

        while abs((self.calc_sig_max(bgk_K, gp, i, j)/snr)-1) > e:
            print("Relation detector: "+str(abs(self.calc_sig_max(bgk_K, gp, i, j)/snr-1)))
            bgk_K*=snr/self.calc_sig_max(bgk_K, gp, i, j)
            print("New K: "+ str(bgk_K))


    def calc_sig_max(self, bgk_K, gp, i, j):
        siglist=[]
        for det in det_list:
            gp.response_generator[det][i, j] = {"generator" : DispersionSpectrumLike.from_function(det, source_function=gp.spectrum_matrix[i, j], background_function=Powerlaw(K=bgk_K), response=gp.response[det])}
            siglist.append(gp.response_generator[det][i,j]['generator'].significance)
        print("Sigmax: " + str(max(siglist)))
        return max(siglist)

    def load_DRM_spectrum(self):
        '''
        Load saved PHA files from folder saved_pha in Simulation grid
        '''
        i_list = []
        j_list = []
        dirs = 0
        for _, dirnames, filenames in os.walk("saved_pha/"):
            for filename in filenames:
                i_list.append(filename.split("_")[1])
                j_list.append(filename.split("_")[2][0])
            dirs += len(dirnames)

        assert len(self.grid) == dirs, "Number of gridpoints do not coincide"
        assert int(max(i_list)) == self.spectrum_dimension[0]-1 and int(
            max(j_list)) == self.spectrum_dimension[1]-1, "Dimensions do not coincide"

        os.chdir("saved_pha/")
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

                        gp.response_generator[det][i, j] = {'generator' : OGIPLike(
                            gp.name+"_"+file_name, observation=file_path+".pha", background=file_path+"_bak.pha", response=file_path+".rsp", spectrum_number=1)}
        os.chdir("../")


class GridPoint():

    '''
    One point in the simulation grid.
    '''

    def __init__(self, name, coord, dim):
        self.name = name
        self.coord = coord
        self.dim = dim
        self.j2000 = None
        self.response = dict()
        self.response_generator = dict()

    def generate_am_spectrum(self, i_max, i_min, c_max, c_min, K):
        """
        Compute sample cutoff powerlaw spectra
        spectrum_matrix:


        """
        n = self.dim[0]
        m = self.dim[1]

        self.spectrum_matrix = np.empty(self.dim, dtype=classmethod)

        self.value_matrix_string = np.empty(self.dim, dtype='U24')

        self.value_matrix = np.empty(
            self.dim, dtype=[('K', 'f8'), ('xc', 'f8'), ('index', 'f8')])
        ''' Array with dimension self.dim
        Each cell has structure [K,xc,index]
        '''
        index = np.linspace(i_min, i_max, n)
        cutoff = np.linspace(c_min, c_max, m)
        i = 0
        # source=PointSource()
        for index_i in index:
            j = 0
            for cutoff_i in cutoff:
                self.spectrum_matrix[i, j] = astromodels.Band_Calderone(F=K)
                #self.spectrum_matrix[i, j] = astromodels.Cutoff_powerlaw(
                #    K=K, index=index_i, xc=cutoff_i, piv=100.)
                # self.value_matrix_string[i, j] = u"Index=" + \
                #     unicode(round(index_i, 3))+u";Cutoff="+unicode(cutoff_i)
                # self.value_matrix[i, j]["K"] = K
                # self.value_matrix[i, j]["xc"] = cutoff_i
                # self.value_matrix[i, j]["index"] = index_i
                # j += 1
            i += 1
        print(self.value_matrix)

    def add_j2000(self, sat_coord, sat_quat, time=2.):
        """
        Calculate the corresponding Ra and Dec coordinates
        for the already given coordinate in the Fermi-Frame

        final_frame:
        doesnt matter as gbm_frame.gbm_to_j2000 outputs only ICRS
        """
        x, y, z = sat_coord
        q1, q2, q3, q4 = sat_quat

        frame = GBMFrame(sc_pos_X=x, sc_pos_Y=y, sc_pos_Z=z, quaternion_1=q1, quaternion_2=q2, quaternion_3=q3, quaternion_4=q4,
                         SCX=self.coord[0]*u.km, SCY=self.coord[1]*u.km, SCZ=self.coord[2]*u.km, representation='cartesian')
        # Muessen die Punkte ins unendliche projiziert werden?
        icrsdata = gbm_frame.gbm_to_j2000(frame, coord.ICRS)
        self.j2000 = icrsdata
        self.ra = self.j2000.ra.degree
        self.dec = self.j2000.dec.degree

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

    def update_coord(self, new_coord):
        self.coord = new_coord
