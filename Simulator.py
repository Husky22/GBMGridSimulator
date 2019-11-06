import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astromodels
from threeML import * 
import astropy.units as u
import astropy.coordinates as coord
from gbmgeometry import PositionInterpolator, GBM, GBMFrame, gbm_frame
from astropy.coordinates import BaseCoordinateFrame, Attribute, RepresentationMapping
from astropy.coordinates import frame_transform_graph
from IPython.display import HTML, display
from tabulate import tabulate


class Simulator():
    """
    Fermi GBM Simulator

    """

    def __init__(self,source_number, spectrum_matrix_dimensions ):
        self.N=source_number
        self.spectrum_dimension=spectrum_matrix_dimensions
        self.grid=None
        self.j2000_generate=False
        self.indexrange=None
        self.energyrange=None


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
        A modified better version of the fibonacci algorithm
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






    def setup(self,irange=[-1.5,-1],erange=[100,400],algorithm='ModifiedFibonacci'):
        '''
        Setup grid and spectrum matrices
        '''

        self.indexrange=irange
        self.energyrange= erange


        if algorithm=='Fibonacci':
            self.grid = self.fibonacci_sphere()
            if self.j2000_generate==True:
                self.generate_j2000(self.trigfile)



        elif algorithm=='ModifiedFibonacci':
            self.grid = self.voronoi_sphere()
            if self.j2000_generate==True:
                self.generate_j2000(self.trigfile)

        for point in self.grid:
            point.generate_spectrum(i_min=float(min(irange)),i_max=float(max(irange)),e_min=float(min(erange)),e_max=float(max(erange)))

    

    def generate_j2000(self,trigdat,final_frame=coord.FK5):
        '''
        Calculate Ra and Dec Values
        '''
        self.trigfile=trigdat
        try:
            for gp in self.grid:
                gp.add_j2000(trigdat,final_frame=final_frame)
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
        pointlist=[]
        for point in self.grid:
            pointlist.append(point.coord)
        return pointlist




class GridPoint():

    def __init__(self,coord,dim):
        self.coord = coord
        self.dim=dim
        self.j2000 = None


    def generate_spectrum(self,i_max,i_min,e_max,e_min):
        """ Compute sample cutoff powerlaw spectra
        """
        n=self.dim[0]
        m=self.dim[1]
        self.spectrum_matrix=np.empty(self.dim,dtype=classmethod)
        self.value_matrix=np.empty(self.dim,dtype='U24')
        index=np.linspace(i_min,i_max,n)
        epmax=np.linspace(e_min,e_max,m)
        i=0
        # source=PointSource()
        for index_i in index:
            j=0
            for epmax_i in epmax:
                self.spectrum_matrix[i,j]= astromodels.Cutoff_powerlaw(K=epmax_i,index=index_i)
                self.value_matrix[i,j]=u"Index="+unicode(round(index_i,3))+u" Energy="+unicode(epmax_i)
                j+=1
            i+=1

    def add_j2000(self,trigdat,time=0.,final_frame=coord.FK5):

        """
        Calculate the corresponding Ra and Dec coordinates
        for the already given coordinate in the Fermi-Frame

        final_frame:
              coord.FK5
              coord.
        """

        position_interpolator= PositionInterpolator(trigdat=trigdat)
        fermi=GBM(position_interpolator.quaternion(time),
                  position_interpolator.sc_pos(time) * u.km)

        x,y,z=position_interpolator.sc_pos(time)
        q1,q2,q3,q4=position_interpolator.quaternion(time)
        frame=GBMFrame(sc_pos_X=x,sc_pos_Y=y,sc_pos_Z=z,quaternion_1=q1,quaternion_2=q2,quaternion_3=q3,quaternion_4=q4,SCX=self.coord[0]*u.km,SCY=self.coord[1]*u.km,SCZ=self.coord[2]*u.km,representation='cartesian')
        # Muessen die Punkte ins unendliche projiziert werden?
        icrsdata=gbm_frame.gbm_to_j2000(frame,final_frame)
        self.j2000=icrsdata

    def show(self):
        print("GBM Cartesian Coordinates: "+ str(self.coord))
        if self.j2000==None:
            print("RA and DEC not calculated yet! Use generate_j2000 function of Simulator to do so.")
        else:
            print("RA: "+ str(self.j2000.ra)+ " \nDEC: "+ str(self.j2000.dec))
        display(HTML(tabulate(self.value_matrix,tablefmt='html',headers=range(self.dim[1]),showindex='always'))) 

    def update_coord(self,new_coord):
        self.coord=new_coord









