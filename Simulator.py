import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astromodels
from threeML import *
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion, ChannelSet, BinnedSpectrum
from threeML.utils.OGIP.response import OGIPResponse
import astropy.units as u
import astropy.coordinates as coord
from gbmgeometry import PositionInterpolator, GBM, GBMFrame, gbm_frame
from astropy.coordinates import BaseCoordinateFrame, Attribute, RepresentationMapping
from astropy.coordinates import frame_transform_graph
from IPython.display import HTML, display
from tabulate import tabulate
import gbm_drm_gen as drm
from glob import glob
import dill


class Simulator():

    """
    Fermi GBM Simulator

    """

    def __init__(self,source_number, spectrum_matrix_dimensions,trigfile):
        self.N=source_number
        self.spectrum_dimension=spectrum_matrix_dimensions
        self.grid=None
        self.j2000_generate=False
        self.indexrange=None
        self.cutoffrange=None
        self.trigfile=trigfile


    def fibonacci_sphere(self,randomize=True):
        """
        The standard algorithm for isotropic point distribution on a sphere based on the fibonacci-series
        """
        samples=self.N
        rnd=1.
        if randomize:
            rnd=random.random()*samples
            points=[]
            offset=2./samples
            increment=math.pi*(3.-math.sqrt(5.))

            for i in range(samples):
                y=((i*offset)-1)+(offset/2)
                r=math.sqrt(1-y**2)
                phi=((i+rnd)%samples)*increment
                x=math.cos(phi)*r
                z=math.sin(phi)*r
                points.append(GridPoint([x,y,z],self.spectrum_dimension))
            return np.array(points)


    def voronoi_sphere(self):

        """
        !!BROKEN!!
        Should have been a modified better version of the fibonacci algorithm
        """
        samples=self.N
        points=[]
        gr = (1+np.sqrt(5))/2
        e=11/2
        # Sequence on [0,1]^2
        for i in range(samples-2):
            if i==0:
                t1=0
                t2=0
            elif i==samples-2:
                t1=1
                t2=0

            else:
                t1=((i+e+0.5)/(samples+2*e))
                t2=(i/gr)


            # Spherical area conserving projection
            p1=np.arccos(2 * t1-1)-np.pi/2
            p2=2*np.pi * t2

            # Transformation to cartesian
            x=np.cos(p1)*np.cos(p2)
            y=np.cos(p1)*np.sin(p2)
            z=np.sin(p1)

            points.append(GridPoint([x,y,z],self.spectrum_dimension))

        return np.array(points)

    def coulomb_refining(self,Nsteps,dt=0.1):
        '''
        Refine your Fibonacci Lattice with coulomb (inverse square) repulsion simulation
        ODE solver is velocity verlet

        Parameters:
        Nsteps: Number of simulation steps
        dt: stepsize
        '''
        # get array of coordinates from GridPoint array
        particles=self.get_coords_from_gridpoints()

        velocities=np.zeros([3,len(particles)])
        steps=range(Nsteps)

        def force_law(pos1,pos2):
            dist=np.linalg.norm(pos1-pos2,2)
            return (pos1-pos2)/(dist**3)

        oldparticles=particles
        oldparticles_temp=particles
        it=0
        for step in steps:
            i=0
            for particle in particles:
                distlist=[]
                otherparticles=np.delete(particles,i,axis=0)
                force=np.zeros(3)
                for otherparticle in otherparticles:

                    distance=np.linalg.norm(particle-otherparticle)
                    distlist.append(distance)


                    force=force+force_law(particle,otherparticle)
                normalcomponent=particle/np.linalg.norm(particle)
                force=force-np.dot(normalcomponent,force)*normalcomponent

                if np.amin(distlist)>1:
                    force*=3

                if it==0:
                    newp=particle+0.5*force*dt**2
                    particles[i]=newp/np.linalg.norm(newp,2)
                else:
                    oldparticles_temp[i]=particles[i]
                    newp=2*particle-oldparticles[i]+force*dt**2
                    particles[i]=newp/np.linalg.norm(newp,2)
                    oldparticles[i]=oldparticles_temp[i]
                i+=1
            it+=1

        for i in range(len(self.grid)):
            self.grid[i].update_coord(particles[i])


    def setup(self,irange=[-1.5,-1],crange=[100,400],K=50,algorithm='Fibonacci',background_function=Powerlaw(K=1.,index=-1.5)):
        '''
        Setup grid, spectrum matrices and the the background function
        '''

        self.indexrange=irange
        self.cutoffrange= crange
        self.background=background_function


        if algorithm=='Fibonacci':
            self.grid = self.fibonacci_sphere()
            if self.j2000_generate==True:
                self.generate_j2000(self.trigfile)



        elif algorithm=='ModifiedFibonacci':
            self.grid = self.voronoi_sphere()
            if self.j2000_generate==True:
                self.generate_j2000(self.trigfile)

        for point in self.grid:
            point.generate_spectrum(i_min=float(min(irange)),i_max=float(max(irange)),c_min=float(min(crange)),c_max=float(max(crange)),K=K)

    
    def generate_j2000(self,time=0.):
        '''
        Calculate Ra and Dec Values
        '''
        position_interpolator= PositionInterpolator(trigdat=self.trigfile)
        # fermi=GBM(position_interpolator.quaternion(time),
        #           position_interpolator.sc_pos(time) * u.km)
        self.sat_coord=position_interpolator.sc_pos(time)
        self.sat_quat=position_interpolator.quaternion(time)
        try:
            for gp in self.grid:
                gp.add_j2000(self.sat_coord,self.sat_quat)
            self.j2000_generate=True
        except:
            print("Error! Is trigdat path correct?")



    def grid_plot(self):
        '''
        Visualize Grid
        '''
        ralist=[]
        declist=[]
        for point in self.grid:
            ralist.append(point.j2000.ra)
            declist.append(point.j2000.dec)
        icrsdata=coord.SkyCoord(ra=ralist*u.degree,dec=declist*u.degree,frame=coord.ICRS)
        plt.subplot(111,projection='aitoff')
        plt.grid(True)
        plt.scatter(icrsdata.ra.wrap_at('180d').radian,icrsdata.dec.radian)


    def get_coords_from_gridpoints(self):
        '''
        Returns python list of all x,y,z coordinates of the points
        '''
        pointlist=[]
        for point in self.grid:
            pointlist.append(point.coord)
        return pointlist

    def generate_TRIG_spectrum(self,trigger="191017391"):
        '''
        Generates DRMs for all GridPoints for all detectors and folds the given spectra matrices
        through it so that we get a simulated physical photon count spectrum

        It uses sample TRIGDAT file

        Generates for every GridPoint:
        response
        response_generator

        '''

        det_list=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        self.det_rsp=dict()
        os.chdir('rawdata/'+trigger)
        for det in det_list:
            rsp = drm.drmgen_trig.DRMGenTrig(self.sat_quat,self.sat_coord,det_list.index(det),tstart=0.,tstop=2.,time=0.)

            self.det_rsp[det] = rsp

        for gp in self.grid:
            ra, dec = gp.ra, gp.dec
            for det in det_list:
                gp.response[det]=self.det_rsp[det].to_3ML_response(ra,dec)
                gp.response_generator[det]=np.empty(gp.dim,dtype=classmethod)
                i=0
                for i in range(np.shape(gp.spectrum_matrix)[0]):
                    j=0
                    for j in range(np.shape(gp.spectrum_matrix)[0]):
                        gp.response_generator[det][i,j]=DispersionSpectrumLike.from_function(det,source_function=gp.spectrum_matrix[i,j],background_function=self.background,response=gp.response[det])
        os.chdir('../../')

    def generate_spectrum_DRMgiven(self,trigger="191017391"):
        '''
        Test for Error finding in DRM generation


        Generates for every GridPoint:
        response
        response_generator

        '''

        det_list=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        self.det_rsp=dict()
        os.chdir('rawdata/'+trigger)
        for det in det_list:
            rsp = drm.drmgen_trig.DRMGenTrig(self.sat_quat,self.sat_coord,det_list.index(det),tstart=0.,tstop=2.,time=0.)

            self.det_rsp[det] = rsp

        for gp in self.grid:
            ra, dec = gp.ra, gp.dec
            for det in det_list:
                gp.response[det]=OGIPResponse(glob('glg_cspec_'+det+'_bn'+trigger+'_v0*.rsp')[0])
                gp.response_generator[det]=np.empty(gp.dim,dtype=classmethod)
                i=0
                for i in range(np.shape(gp.spectrum_matrix)[0]):
                    j=0
                    for j in range(np.shape(gp.spectrum_matrix)[0]):
                        gp.response_generator[det][i,j]=DispersionSpectrumLike.from_function(det,source_function=gp.spectrum_matrix[i,j],background_function=self.background,response=gp.response[det])
        os.chdir('../../')

    def generate_spectrum_from_function(self,det,source_function,background_function,response,ra,dec):
        '''
        An alternative Spectrum Generator derived from older 3ML version
        '''

        channel_set= ChannelSet.from_instrument_response(response)
        energy_min, energy_max = channel_set.bin_stack.T
        fake_data=np.ones(len(energy_min))
        n_energies=response.ebounds.shape[0]-1
        observation=DispersionSpectrumLike._build_fake_observation(fake_data,channel_set,None,None,True,response=response)

        tmp_background=BinnedSpectrum(np.ones(n_energies), exposure=1.,ebounds=response.ebounds,count_errors=None,sys_errors=None,quality=None,scale_factor=1.,is_poisson=True,mission='fake_mission',instrument='fake_instrument',tstart=0.,tstop=2.)
        background_gen=SpectrumLike('generatorbkg',tmp_background,None,verbose=False)
        pts_background=PointSource('fake_background',0.0,0.0,background_function)
        background_model=Model(pts_background)
        background_gen.set_model(background_model)
        sim_background=background_gen.get_simulated_dataset('fake')
        background=sim_background._observed_spectrum
        speclike_gen=DispersionSpectrumLike('generator',observation,background=background)
        pts=PointSource('fake',ra,dec,source_function)
        model1=Model(pts)
        speclike_gen.set_model(model1)
        return speclike_gen.get_simulated_dataset(det)


    def generate_TRIG_spectrum2(self,trigger="191017391"):
        '''
        Generates DRMs for all GridPoints for all detectors and folds the given spectra matrices
        through it so that we get a simulated physical photon count spectrum

        It uses sample TTE,TRIGDAT and CSPEC files to generate the DRMs

        Generates for every GridPoint:
        response
        response_generator

        '''

        det_list=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        self.det_rsp=dict()
        os.chdir('rawdata/'+trigger)
        for det in det_list:
            rsp = drm.drmgen_trig.DRMGenTrig(self.sat_quat,self.sat_coord,det_list.index(det),tstart=0.,tstop=2.,time=0.)

            self.det_rsp[det] = rsp

        for gp in self.grid:
            ra, dec = gp.ra, gp.dec
            for det in det_list:
                gp.response[det]=self.det_rsp[det].to_3ML_response(ra,dec)
                gp.response_generator[det]=np.empty(gp.dim,dtype=classmethod)
                i=0
                for i in range(np.shape(gp.spectrum_matrix)[0]):
                    j=0
                    for j in range(np.shape(gp.spectrum_matrix)[0]):
                        gp.response_generator[det][i,j]=self.generate_spectrum_from_function(det,source_function=gp.spectrum_matrix[i,j],background_function=self.background,response=gp.response[det],ra=gp.ra,dec=gp.dec)
        os.chdir('../../')


    def generate_DRM_spectrum(self,trigger="191017391",save=False):
        
        '''
        Generates DRMs for all GridPoints for all detectors and folds the given spectra matrices
        through it so that we get a simulated physical photon count spectrum

        It uses sample TTE,TRIGDAT and CSPEC files to generate the DRMs

        Generates for every GridPoint:
        response
        response_generator

        save: Save count spectra as pdf

        '''

        det_list=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        self.det_rsp=dict()
        os.chdir('rawdata/'+trigger)
        for det in det_list:
            rsp = drm.DRMGenTTE(tte_file=glob('glg_tte_'+det+'_bn'+trigger+'_v0*.fit.gz')[0],trigdat=glob('glg_trigdat_all_bn'+trigger+'_v0*.fit')[0],mat_type=2,cspecfile=glob('glg_cspec_'+det+'_bn'+trigger+'_v0*.pha')[0])

            self.det_rsp[det] = rsp

        for gp in self.grid:
            ra, dec = gp.ra, gp.dec
            for det in det_list:
                gp.response[det]=self.det_rsp[det].to_3ML_response(ra,dec)
                gp.response_generator[det]=np.empty(gp.dim,dtype=classmethod)
                i=0
                for i in range(np.shape(gp.spectrum_matrix)[0]):
                    j=0
                    for j in range(np.shape(gp.spectrum_matrix)[0]):
                        gp.response_generator[det][i,j]=DispersionSpectrumLike.from_function(det,source_function=gp.spectrum_matrix[i,j],background_function=self.background,response=gp.response[det])
                        if save==True:
                            gp.response_generator[det][i,j].view_count_spectrum().savefig("../../saves/"+det+"_"+str(i)+"_"+str(j)+".pdf")

        os.chdir('../../')

    def save(self,fname):
        np.save(fname,self.grid,allow_pickle=False)

    def load(self,fname):
        self.grid=np.load(fname,allow_pickle=True)


class GridPoint():

    '''
    One point in the simulation grid.
    '''

    def __init__(self,coord,dim):
        self.coord = coord
        self.dim=dim
        self.j2000 = None
        self.response=dict()
        self.response_generator=dict()


    def generate_spectrum(self,i_max,i_min,c_max,c_min,K):
        """
        Compute sample cutoff powerlaw spectra
        """
        n=self.dim[0]
        m=self.dim[1]
        self.spectrum_matrix=np.empty(self.dim,dtype=classmethod)
        self.value_matrix_string=np.empty(self.dim,dtype='U24')
        self.value_matrix=np.empty(self.dim,dtype=[('K','f8'),('xc','f8'),('index','f8')])
        index=np.linspace(i_min,i_max,n)
        cutoff=np.linspace(c_min,c_max,m)
        i=0
        # source=PointSource()
        for index_i in index:
            j=0
            for cutoff_i in cutoff:
                self.spectrum_matrix[i,j]= astromodels.Cutoff_powerlaw(K=K,index=index_i,xc=cutoff_i)
                self.value_matrix_string[i,j]=u"Index="+unicode(round(index_i,3))+u";Cutoff="+unicode(cutoff_i)
                self.value_matrix[i,j]["K"]=K
                self.value_matrix[i,j]["xc"]=cutoff_i
                self.value_matrix[i,j]["index"]=index_i
                j+=1
            i+=1
        print(self.value_matrix)


    def add_j2000(self,sat_coord,sat_quat,time=2.):

        """
        Calculate the corresponding Ra and Dec coordinates
        for the already given coordinate in the Fermi-Frame

        final_frame:
        doesnt matter as gbm_frame.gbm_to_j2000 outputs only ICRS
        """
        x,y,z=sat_coord
        q1,q2,q3,q4=sat_quat

        frame=GBMFrame(sc_pos_X=x,sc_pos_Y=y,sc_pos_Z=z,quaternion_1=q1,quaternion_2=q2,quaternion_3=q3,quaternion_4=q4,SCX=self.coord[0]*u.km,SCY=self.coord[1]*u.km,SCZ=self.coord[2]*u.km,representation='cartesian')
        # Muessen die Punkte ins unendliche projiziert werden?
        icrsdata=gbm_frame.gbm_to_j2000(frame,coord.ICRS)
        self.j2000=icrsdata
        self.ra=self.j2000.ra.degree
        self.dec=self.j2000.dec.degree


    def show(self):
        '''
        Returns coordinates in cartesian and ICRS and a table with the generated sample spectrum parameters
        '''
        print("GBM Cartesian Coordinates: "+ str(self.coord))
        if self.j2000==None:
            print("RA and DEC not calculated yet! Use generate_j2000 function of Simulator to do so.")
        else:
            print("RA: "+ str(self.j2000.ra)+ " \nDEC: "+ str(self.j2000.dec))
        display(HTML(tabulate(self.value_matrix_string,tablefmt='html',headers=range(self.dim[1]),showindex='always'))) 


    def update_coord(self,new_coord):
        self.coord=new_coord









